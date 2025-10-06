import os
import json
import time
import argparse
import shutil
from importlib import reload
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset_lstm import CaptionFeatureDataset, collate_fn, load_json
from model_lstm import EncoderIdentity, DecoderRNN

# BLEU utilities: try nltk, else simple fallback
def try_import_nltk():
	try:
		import nltk
		from nltk.translate.bleu_score import corpus_bleu
		return corpus_bleu
	except Exception:
		return None

corpus_bleu = try_import_nltk()

def compute_bleu(references, hypotheses):
	"""
	references: list of reference token lists (for each candidate)
	hypotheses: list of candidate token lists
	"""
	if corpus_bleu is not None:
		return corpus_bleu(references, hypotheses)
	# fallback simple BLEU-4 approximation (unigram precision only)
	def unigram_precision(refs, hyp):
		ref_tokens = set()
		for r in refs:
			ref_tokens.update(r)
		if len(hyp)==0: return 0.0
		match = sum(1 for t in hyp if t in ref_tokens)
		return match / len(hyp)
	ps = [unigram_precision(r, h) for r,h in zip(references, hypotheses)]
	return sum(ps)/len(ps)

def save_model(state, path):
	torch.save(state, path)

def load_vocab(vocab_path):
	with open(vocab_path, "r", encoding="utf-8") as f:
		return json.load(f)

def train_loop(cfg):
	device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

	# load datasets
	train_ds = CaptionFeatureDataset(cfg['train_sequences'], cfg['features_dir'], cfg['vocab_path'])
	val_ds = CaptionFeatureDataset(cfg['val_sequences'], cfg['features_dir'], cfg['vocab_path'])
	# dataloaders
	train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2)
	val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2)

	# vocab size and indices
	word2idx = load_vocab(cfg['vocab_path'])
	vocab_size = len(word2idx)
	start_token = word2idx.get("<START>", 2)
	pad_idx = word2idx.get("<PAD>", 0)

	# model
	encoder = EncoderIdentity(feat_dim=cfg['feat_dim'], embed_dim=cfg['feat_embed_dim']).to(device)
	decoder = DecoderRNN(embed_dim=cfg['embed_dim'], hidden_dim=cfg['hidden_dim'],
						 vocab_size=vocab_size, num_layers=cfg['num_layers'],
						 dropout=cfg['dropout'], feat_embed_dim=cfg['feat_embed_dim']).to(device)

	# combine params
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer = torch.optim.Adam(params, lr=cfg['lr'])
	# scheduler reduces LR when val BLEU plateaus
	scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

	criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

	best_score = float("-inf")
	patience_counter = 0

	# history dict
	history = {
		"train_loss": [],
		"val_loss": [],
		"val_bleu": []
	}

	save_idx = 0

	if 'resume_path' in cfg and os.path.isfile(cfg['resume_path']):
		print("Loading checkpoint:", cfg['resume_path'])
		state = torch.load(cfg['resume_path'], map_location=device)
		encoder.load_state_dict(state['encoder_state'])
		decoder.load_state_dict(state['decoder_state'])
		optimizer.load_state_dict(state['optimizer_state'])
		start_epoch = state.get('epoch', 0) + 1
		best_score = state.get('val_bleu', float('-inf'))
		print(f"Resuming from epoch {start_epoch}, best val_bleu={best_score:.4f}")
	else:
		start_epoch = 1


	for epoch in range(start_epoch, cfg['epochs']+1):
		decoder.train()
		encoder.train()
		train_loss = 0.0
		it = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
		for feats, seqs, _ in it:
			feats = feats.to(device)
			seqs = seqs.to(device)
			optimizer.zero_grad()
			logits = decoder(encoder(feats), seqs)
			B, Lm1, V = logits.size()
			loss = criterion(logits.view(B*Lm1, V), seqs[:,1:].reshape(-1))
			loss.backward()
			# gradient clipping
			torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
			optimizer.step()
			train_loss += loss.item()
			it.set_postfix(loss=loss.item())
		avg_train_loss = train_loss / len(train_loader)

		decoder.eval()
		encoder.eval()
		val_loss = 0.0
		references = []
		hypotheses = []
		with torch.no_grad():
			for feats, seqs, _ in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
				feats = feats.to(device)
				seqs = seqs.to(device)
				enc = encoder(feats)
				logits = decoder(enc, seqs)
				B, Lm1, V = logits.size()
				loss = criterion(logits.view(B*Lm1, V), seqs[:,1:].reshape(-1))
				val_loss += loss.item()

				# greedy decode
				preds = decoder.greedy_decode(enc, start_token=start_token, max_len=cfg['max_len'])
				for i in range(preds.size(0)):
					hyp = preds[i].cpu().tolist()
					if word2idx.get("<END>") in hyp:
						endpos = hyp.index(word2idx.get("<END>"))
						hyp = hyp[:endpos]
					ref_seq = seqs[i].cpu().tolist()
					if word2idx.get("<START>") in ref_seq:
						ref = ref_seq[1:]
					else:
						ref = ref_seq
					if word2idx.get("<END>") in ref:
						endpos = ref.index(word2idx.get("<END>"))
						ref = ref[:endpos]
					references.append([ref])
					hypotheses.append(hyp)

		avg_val_loss = val_loss / len(val_loader)
		try:
			score = compute_bleu(references, hypotheses)
		except Exception:
			score = 0.0

		# logging
		print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_bleu={score:.4f}")

		# update history
		history["train_loss"].append(avg_train_loss)
		history["val_loss"].append(avg_val_loss)
		history["val_bleu"].append(score)

		# step scheduler on validation BLEU
		try:
			scheduler.step(score)
		except Exception:
			# if scheduler fails for some reason, continue
			pass

		# checkpoint on improvement
		if score > best_score:
			best_score = score
			patience_counter = 0
			save_path = os.path.join(cfg['save_dir'], f"best_model_epoch{epoch}_bleu{score:.4f}.pt")
			os.makedirs(cfg['save_dir'], exist_ok=True)
			# save optimizer state and epoch for resuming
			state = {
				'epoch': epoch,
				'encoder_state': encoder.state_dict(),
				'decoder_state': decoder.state_dict(),
				'optimizer_state': optimizer.state_dict(),
				'val_loss': avg_val_loss,
				'val_bleu': score,
				'word2idx': load_json(cfg['vocab_path'])
			}
			save_model(state, save_path)
			save_idx += 1
			if save_idx % 5 == 0:
				try:
					save_model(state, os.path.join("/content/drive/MyDrive/captioneer/models", f"best_model_epoch{epoch}_bleu{score:.4f}.pt"))
					print(f"Epoch {epoch}: checkpoint copied to Drive")
				except Exception as e:
					print("Warning: failed to copy checkpoint to Drive:", e)
			# print lr info
			cur_lr = optimizer.param_groups[0]['lr']
			print(f"Saved improved model to {save_path} (lr={cur_lr:.6g})")
		else:
			patience_counter += 1
			if patience_counter >= cfg['patience']:
				print("Early stopping triggered.")
				break

	# return history at the end
	return history

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_seq", type=str, required=True)
	parser.add_argument("--val_seq", type=str, required=True)
	parser.add_argument("--features_dir", type=str, required=True)
	parser.add_argument("--vocab", type=str, required=True)
	parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/captioneer/models")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--patience", type=int, default=3)
	parser.add_argument("--embed_dim", type=int, default=256)
	parser.add_argument("--hidden_dim", type=int, default=512)
	parser.add_argument("--num_layers", type=int, default=1)
	parser.add_argument("--dropout", type=float, default=0.5)
	parser.add_argument("--feat_dim", type=int, default=2048)
	parser.add_argument("--feat_embed_dim", type=int, default=512)
	parser.add_argument("--max_len", type=int, default=30)
	parser.add_argument("--resume_path", type=str, required=False)

	args = parser.parse_args()
	cfg = {
		'train_sequences': args.train_seq,
		'val_sequences': args.val_seq,
		'features_dir': args.features_dir,
		'vocab_path': args.vocab,
		'save_dir': args.save_dir,
		'device': args.device,
		'batch_size': args.batch_size,
		'epochs': args.epochs,
		'lr': args.lr,
		'patience': args.patience,
		'embed_dim': args.embed_dim,
		'hidden_dim': args.hidden_dim,
		'num_layers': args.num_layers,
		'dropout': args.dropout,
		'feat_dim': args.feat_dim,
		'feat_embed_dim': args.feat_embed_dim,
		'max_len': args.max_len,
		'resume_path': ""
	}
	train_loop(cfg)
