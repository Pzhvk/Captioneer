import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

def load_json(path):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

class CaptionFeatureDataset(Dataset):
	"""
	Dataset that returns (image_feature_tensor, caption_tensor).
	The sequences files are expected to be JSON mapping image_name -> list of padded sequences (lists of ints).
	We flatten so each (image, caption) pair is a sample.
	"""
	def __init__(self, sequences_json, features_dir, vocab_json=None):
		self.sequences = load_json(sequences_json)  # image -> list of seqs
		self.features_dir = features_dir
		self.items = []  # list of (image_name, seq_list)
		for img, caps in self.sequences.items():
			for cap in caps:
				self.items.append((img, cap))
		self.vocab = None
		if vocab_json:
			self.vocab = load_json(vocab_json)

	def __len__(self):
		return len(self.items)

	def __getitem__(self, idx):
		img_name, seq = self.items[idx]
		feat_path = os.path.join(self.features_dir, img_name.replace('.jpg', '.npy'))
		feat = np.load(feat_path)
		feat = torch.from_numpy(feat).float()
		seq = torch.tensor(seq, dtype=torch.long)
		if self.vocab is not None:
			vocab_size = len(self.vocab)
			unk_idx = self.vocab.get("<UNK>", vocab_size - 1)
			seq = torch.where(seq >= vocab_size, torch.tensor(unk_idx, dtype=torch.long), seq)
			seq = torch.where(seq < 0, torch.tensor(unk_idx, dtype=torch.long), seq)
		return feat, seq, img_name

def collate_fn(batch):
	"""
	batch: list of (feat, seq, img_name)
	feat: (2048,), seq: (max_len,)
	return batched tensors
	"""
	feats = torch.stack([b[0] for b in batch], dim=0)
	seqs = torch.stack([b[1] for b in batch], dim=0)
	img_names = [b[2] for b in batch]
	return feats, seqs, img_names
