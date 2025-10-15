import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer
import math
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    """
    ResNet50 backbone that returns spatial features.
    features: (B, N, feat_embed_dim)
    """
    def __init__(self, feat_embed_dim=768, train_backbone=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone

        # Project ResNet's 2048 features to the decoder's dimension
        self.feat_proj = nn.Linear(2048, feat_embed_dim)

    def forward(self, images):
        x = self.backbone(images)
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)
        feats = self.feat_proj(x)
        return feats

class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class DecoderWithTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, decoder_layers, dim_feedforward, dropout, device, word2idx):
        super().__init__()
        self.device = device
        self.word_embedding = self._create_distilbert_embedding_layer(word2idx, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _create_distilbert_embedding_layer(self, word2idx, embed_dim):
        """Initializes an embedding layer with weights from distilbert-base-uncased."""
        print("Loading DistilBERT model to extract word embeddings...")
        distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        distilbert_embeddings = distilbert.embeddings.word_embeddings
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        vocab_size = len(word2idx)
        embedding_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx.get('<PAD>', 0))
        
        # Initialize with random values first
        embedding_layer.weight.data.uniform_(-0.1, 0.1)

        print(f"Initializing embedding layer for vocabulary of size {vocab_size}...")
        
        for word, idx in word2idx.items():
            # Ensure the index is within the bounds of our new embedding layer
            if idx >= vocab_size:
                continue

            # Special handling for our custom <UNK> token, map it to BERT's [UNK]
            token_to_lookup = '[UNK]' if word == '<UNK>' else word
            
            if token_to_lookup in distilbert_tokenizer.vocab:
                bert_idx = distilbert_tokenizer.vocab[token_to_lookup]
                embedding_layer.weight.data[idx] = distilbert_embeddings.weight.data[bert_idx]
            else:
                # If the word is not in BERT's vocab, it keeps its random initialization.
                pass
        
        # Explicitly set the padding token embedding to zeros for stability
        pad_idx = word2idx.get('<PAD>', 0)
        embedding_layer.weight.data[pad_idx].zero_()

        print("Custom embedding layer created and initialized with DistilBERT weights.")
        return embedding_layer

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)

    def forward(self, feats, captions):
        """
        feats: Image features from EncoderCNN (B, N, embed_dim) -> acts as 'memory'
        captions: Ground truth captions (B, L)
        """
        # Embed captions and add positional encoding
        captions_embedded = self.word_embedding(captions) * math.sqrt(self.word_embedding.embedding_dim)
        captions_embedded = self.pos_encoder(captions_embedded)
        
        # Create target mask to prevent looking at future words
        tgt_mask = self.generate_square_subsequent_mask(captions.size(1))
        
        output = self.transformer_decoder(tgt=captions_embedded, memory=feats, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def beam_search_decode(self, feats, start_token, end_token, beam_width=3, max_len=30):
        B = feats.size(0)
        batch_predictions = []

        for i in range(B):
            single_feats = feats[i].unsqueeze(0)
            
            beam = [(0.0, [start_token])]
            completed_sequences = []

            for _ in range(max_len):
                new_beam = []
                for log_prob, seq in beam:
                    if seq[-1] == end_token:
                        completed_sequences.append((log_prob, seq))
                        continue
                    
                    input_tensor = torch.tensor([seq], device=self.device)
                    logits = self.forward(single_feats, input_tensor)
                    next_word_logits = logits[:, -1, :]
                    log_probs = F.log_softmax(next_word_logits, dim=-1)

                    top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

                    for k in range(beam_width):
                        new_seq = seq + [top_indices[0, k].item()]
                        new_log_prob = log_prob + top_log_probs[0, k].item()
                        new_beam.append((new_log_prob, new_seq))

                beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

                if not beam:
                    break
            
            if completed_sequences:
                best_seq = sorted(completed_sequences, key=lambda x: x[0], reverse=True)[0][1]
                batch_predictions.append(best_seq)
            elif beam:
                best_seq = sorted(beam, key=lambda x: x[0], reverse=True)[0][1]
                batch_predictions.append(best_seq)
            else: # Fallback for rare cases
                batch_predictions.append([start_token, end_token])
        
        return batch_predictions