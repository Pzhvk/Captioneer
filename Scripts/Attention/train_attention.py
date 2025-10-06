import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nltk
from nltk.translate.bleu_score import corpus_bleu
from torch.amp import GradScaler, autocast

from dataset_attention import CaptionImageDataset, collate_fn, load_json
from model_attention import EncoderCNN, DecoderWithAttention

def compute_bleu(references, hypotheses):
    if corpus_bleu is not None:
        return corpus_bleu(references, hypotheses)

    score = 0.0
    for refs, hyp in zip(references, hypotheses):
        ref_tokens = set()
        for r in refs:
            ref_tokens.update(r)

        if not hyp: continue
        match = sum(1 for t in hyp if t in ref_tokens)
        score += match / len(hyp)
    return score / len(hypotheses) if hypotheses else 0.0


def save_checkpoint(state, path):
    """Saves model and training state."""
    torch.save(state, path)


def evaluate(encoder, decoder, val_loader, criterion, device, word2idx, cfg):
    """Performs validation, calculates loss and BLEU score."""
    decoder.eval()
    encoder.eval()
    val_loss = 0.0
    references_corpus = []
    hypotheses_corpus = []

    start_token = word2idx.get("<START>", 2)
    end_token = word2idx.get("<END>", 3)

    with torch.no_grad():
        for images, seqs_list, _ in tqdm(val_loader, desc=f"Validation", leave=False):
            images = images.to(device)
            first_seqs = torch.stack([s[0] for s in seqs_list]).to(device)

            with autocast(device_type=device.type):
                enc_out, _ = encoder(images)
                logits = decoder(enc_out, first_seqs)
                B, Lm1, V = logits.size()
                loss = criterion(logits.view(B * Lm1, V), first_seqs[:, 1:].reshape(-1))
                val_loss += loss.item()

                preds_list = decoder.beam_search_decode(enc_out, start_token, end_token, max_len=cfg['max_len'])

            for i in range(len(seqs_list)):
                refs = []
                for seq_tensor in seqs_list[i]:
                    ref = seq_tensor.cpu().tolist()
                    ref = ref[1:ref.index(end_token)] if end_token in ref else ref[1:]
                    refs.append(ref)
                references_corpus.append(refs)

                hyp = preds_list[i]
                hyp = hyp[1:hyp.index(end_token)] if end_token in hyp else hyp[1:]
                hypotheses_corpus.append(hyp)

    avg_val_loss = val_loss / len(val_loader)
    bleu_score = compute_bleu(references_corpus, hypotheses_corpus)

    return avg_val_loss, bleu_score


def train_loop(cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    word2idx = load_json(cfg['vocab_path'])
    train_ds = CaptionImageDataset(cfg['train_sequences'], cfg['images_dir'], cfg['vocab_path'])
    val_ds = CaptionImageDataset(cfg['val_sequences'], cfg['images_dir'], cfg['vocab_path'], is_val=True)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn,
                              num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2)

    pad_idx = word2idx.get("<PAD>", 0)

    encoder = EncoderCNN(feat_dim=cfg['feat_dim'], feat_embed_dim=cfg['feat_embed_dim'],
                         train_backbone=cfg.get('train_backbone', False)).to(device)
    decoder = DecoderWithAttention(embed_dim=cfg['embed_dim'], hidden_dim=cfg['hidden_dim'],
                                   vocab_size=len(word2idx), feat_dim=cfg['feat_dim'],
                                   feat_embed_dim=cfg['feat_embed_dim'], attn_dim=cfg.get('attn_dim', 512),
                                   num_layers=cfg['num_layers'], dropout=cfg['dropout']).to(device)

    if cfg.get('train_backbone', False):
        params = list(decoder.parameters()) + list(encoder.parameters())
    else:
        params = list(decoder.parameters())

    optimizer = torch.optim.Adam(params, lr=cfg['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scaler = GradScaler()

    best_score = float("-inf")
    patience_counter = 0
    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], "val_bleu": []}

    if 'resume_path' in cfg and cfg['resume_path'] and os.path.isfile(cfg['resume_path']):
        print("Loading checkpoint:", cfg['resume_path'])
        state = torch.load(cfg['resume_path'], map_location=device)
        encoder.load_state_dict(state.get('encoder_state', {}))
        decoder.load_state_dict(state.get('decoder_state', {}))
        if state.get('optimizer_state'):
            optimizer.load_state_dict(state.get('optimizer_state'))
        start_epoch = state.get('epoch', 0) + 1
        best_score = state.get('val_bleu', float('-inf'))
        print(f"Resuming from epoch {start_epoch}, best val_bleu={best_score:.4f}")

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        decoder.train()
        encoder.train()
        train_loss = 0.0
        it = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=True)

        for images, seqs, _ in it:
            images, seqs = images.to(device), seqs.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                enc_out, _ = encoder(images)
                logits = decoder(enc_out, seqs)
                B, Lm1, V = logits.size()
                loss = criterion(logits.view(B * Lm1, V), seqs[:, 1:].reshape(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            it.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        avg_val_loss, score = evaluate(encoder, decoder, val_loader, criterion, device, word2idx, cfg)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_bleu={score:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_bleu"].append(score)

        scheduler.step(score)

        if score > best_score:
            best_score = score
            patience_counter = 0
            save_path = os.path.join(cfg['save_dir'], f"best_model_epoch{epoch}_bleu{score:.4f}.pt")
            os.makedirs(cfg['save_dir'], exist_ok=True)
            state = {
                'epoch': epoch,
                'encoder_state': encoder.state_dict(),
                'decoder_state': decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_bleu': score,
            }
            save_checkpoint(state, save_path)
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Saved improved model to {save_path} (lr={cur_lr:.6g})")
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print("Early stopping triggered.")
                break

    return history

