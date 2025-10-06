import os
import json
import re
import random
import shutil
from collections import Counter, defaultdict
import pandas as pd

def save_json(obj, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

def read_results_csv(csv_path, image_col="image_name", caption_col="comment"):
	"""Read CSV and return dict: image_name -> list of raw captions."""
	df = pd.read_csv(csv_path)  # standard CSV
	captions = defaultdict(list)
	for _, row in df.iterrows():
		img = str(row[image_col]).strip()
		cap = str(row[caption_col]).strip()
		if cap:
			captions[img].append(cap)
	return dict(captions)

RE_KEEP = re.compile(r"[^a-z0-9'\s]")

def clean_text(text):
	"""Lowercase and remove unwanted characters; keep apostrophes and numbers."""
	if text is None:
		return ""
	s = text.lower().strip()
	s = RE_KEEP.sub('', s)
	s = re.sub(r"\s+", " ", s)
	return s

def clean_captions(raw_captions, max_len=30):
	"""Apply cleaning, add <START>/<END>, drop empty or too-long captions."""
	cleaned = {}
	for img, caps in raw_captions.items():
		out_caps = []
		for c in caps:
			c_clean = clean_text(c)
			tokens = c_clean.split()
			if not tokens:
				continue
			tokens = ["<START>"] + tokens + ["<END>"]
			if len(tokens) <= max_len:
				out_caps.append(" ".join(tokens))
		if out_caps:
			cleaned[img] = out_caps
	return cleaned

def build_vocab(cleaned_captions, vocab_size=5000, special_tokens=("<PAD>", "<UNK>", "<START>", "<END>")):
	"""Return word2idx and idx2word. Most frequent tokens are kept after special tokens."""
	counter = Counter()
	for caps in cleaned_captions.values():
		for sent in caps:
			counter.update(sent.split())
	num_reserved = len(list(special_tokens))
	most_common = [w for w, _ in counter.most_common(max(0, vocab_size - num_reserved))]
	vocab = list(special_tokens) + most_common
	word2idx = {w: i for i, w in enumerate(vocab)}
	idx2word = {i: w for w, i in word2idx.items()}
	return word2idx, idx2word

def tokens_to_indices(sentence, word2idx):
	unk_idx = word2idx.get("<UNK>")
	return [word2idx.get(tok, unk_idx) for tok in sentence.split()]

def pad_sequence(seq, max_len, pad_value=0):
	if len(seq) >= max_len:
		return seq[:max_len]
	return seq + [pad_value] * (max_len - len(seq))

def captions_to_sequences(cleaned_captions, word2idx, max_len):
	"""Convert cleaned captions to padded index sequences."""
	seqs = {}
	for img, caps in cleaned_captions.items():
		out = []
		for c in caps:
			idxs = tokens_to_indices(c, word2idx)
			idxs = pad_sequence(idxs, max_len, pad_value=word2idx.get("<PAD>", 0))
			out.append(idxs)
		if out:
			seqs[img] = out
	return seqs

def split_image_ids(image_ids, ratios=(0.8, 0.1, 0.1), seed=42):
	"""Shuffle (deterministic with seed) and split into train/val/test."""
	assert image_ids, "image_ids is empty"
	r0, r1, r2 = ratios
	assert abs((r0 + r1 + r2) - 1.0) < 1e-6, "ratios must sum to 1.0"
	ids = list(image_ids)
	random.Random(seed).shuffle(ids)
	n = len(ids)
	i0 = int(n * r0)
	i1 = i0 + int(n * r1)
	train = ids[:i0]
	val = ids[i0:i1]
	test = ids[i1:]
	return train, val, test

def save_processed_captions(out_dir, cleaned_captions, word2idx, idx2word, sequences_splits, meta=None):
	os.makedirs(out_dir, exist_ok=True)
	save_json(cleaned_captions, os.path.join(out_dir, "cleaned_captions.json"))
	save_json(word2idx, os.path.join(out_dir, "vocab_word2idx.json"))
	save_json(idx2word, os.path.join(out_dir, "vocab_idx2word.json"))
	for split in ("train", "val", "test"):
		seqs = sequences_splits.get(split, {})
		save_json(seqs, os.path.join(out_dir, f"sequences_{split}.json"))
	if meta is None:
		meta = {}
	save_json(meta, os.path.join(out_dir, "meta.json"))

def copy_folder_to_drive(src, dst):
	"""Copy a local folder to Drive; dst must not already exist."""
	if not os.path.exists(src):
		raise FileNotFoundError(f"Source folder not found: {src}")
	if os.path.exists(dst):
		raise FileExistsError(f"Destination already exists: {dst}")
	shutil.copytree(src, dst)

def preprocess_and_save(csv_path, out_dir, vocab_size=5000, max_len=30, split_ratios=(0.8, 0.1, 0.1), seed=42):
	"""Run the full preprocessing pipeline and save artifacts to out_dir.
	Returns a dict with out_dir and meta info.
	"""
	raw = read_results_csv(csv_path)
	cleaned = clean_captions(raw, max_len=max_len)
	word2idx, idx2word = build_vocab(cleaned, vocab_size)
	sequences = captions_to_sequences(cleaned, word2idx, max_len)

	img_ids = list(sequences.keys())
	train_ids, val_ids, test_ids = split_image_ids(img_ids, split_ratios, seed)

	sequences_splits = {
		"train": {i: sequences[i] for i in train_ids},
		"val": {i: sequences[i] for i in val_ids},
		"test": {i: sequences[i] for i in test_ids}
	}

	meta = {
		"vocab_size": len(word2idx),
		"num_images": len(img_ids),
		"max_len": max_len,
		"split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
	}

	save_processed_captions(out_dir, cleaned, word2idx, idx2word, sequences_splits, meta)
	return {"out_dir": out_dir, "meta": meta}