import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderIdentity(nn.Module):
	"""
	Placeholder encoder that optionally projects 2048-dim features to embed_dim.
	Used because we already have precomputed features with ResNet.
	"""
	def __init__(self, feat_dim=2048, embed_dim=512):
		super().__init__()
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.project = nn.Linear(feat_dim, embed_dim)

	def forward(self, features):
		# features: (B, feat_dim)
		# return projected features and initial hidden for decoder if needed
		enc = self.project(features)
		return enc

class DecoderRNN(nn.Module):
	"""
	Simple Decoder RNN (LSTM) with word embeddings.
	- embed_dim: word embedding size
	- hidden_dim: LSTM hidden size
	- vocab_size: output vocabulary
	"""
	def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.5, feat_embed_dim=512):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
		self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
		self.init_fc = nn.Linear(feat_embed_dim, hidden_dim)  # to init hidden state from image features
		self.init_cell_fc = nn.Linear(feat_embed_dim, hidden_dim)
		self.fc_out = nn.Linear(hidden_dim, vocab_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, features, captions):
		"""
		features: (B, feat_embed_dim)  -- projected features from encoder
		captions: (B, L) -- full padded sequences (with <START> and <END>)
		We compute logits for each time step predicting next token.
		"""
		# prepare inputs: feed embeddings for captions excluding last token
		emb = self.embed(captions[:, :-1])
		emb = self.dropout(emb)
		# initialize hidden state from features
		h0 = self.init_fc(features).unsqueeze(0)
		c0 = self.init_cell_fc(features).unsqueeze(0)
		out, _ = self.lstm(emb, (h0, c0))
		logits = self.fc_out(out)
		return logits

	def greedy_decode(self, features, start_token, max_len=30):
		"""
		Generate captions greedily given features.
		features: (B, feat_embed_dim)
		return: tensor (B, max_len) of predicted token ids (includes start and predicted tokens)
		"""
		B = features.size(0)
		hidden = (self.init_fc(features).unsqueeze(0), self.init_cell_fc(features).unsqueeze(0))
		input_tok = torch.full((B,1), start_token, dtype=torch.long, device=features.device)
		result = []
		for t in range(max_len):
			emb = self.embed(input_tok)
			out, hidden = self.lstm(emb, hidden)
			logit = self.fc_out(out.squeeze(1))
			preds = torch.argmax(logit, dim=-1, keepdim=True)
			result.append(preds)
			input_tok = preds
		preds = torch.cat(result, dim=1)
		return preds
