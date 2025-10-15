import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_image_transform(image_size=224):
    # Resize+center-crop + ImageNet normalization
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class CaptionImageDataset(Dataset):
    """
    Returns (image_tensor, caption_tensor(s), image_name).
    For training (is_val=False), returns one caption per image.
    For validation (is_val=True), returns all 5 captions per image.
    """

    def __init__(self, sequences_json, images_dir, vocab_json=None, transform=None, image_size=224, is_val=False):
        self.sequences = load_json(sequences_json)
        self.images_dir = images_dir
        self.is_val = is_val
        self.items = []
        if self.is_val:
            # For validation, each item is a unique image name
            self.items = list(self.sequences.keys())
        else:
            # For training, flatten image->captions pairs to samples
            for img, caps in self.sequences.items():
                for cap in caps:
                    self.items.append((img, cap))

        self.vocab = None
        if vocab_json:
            self.vocab = load_json(vocab_json)
        self.transform = transform if transform is not None else default_image_transform(image_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.is_val:
            img_name = self.items[idx]
            seqs = self.sequences[img_name]
        else:
            img_name, seq = self.items[idx]
            seqs = [seq]  # Keep it in a list for consistent processing

        img_path = os.path.join(self.images_dir, img_name)
        with Image.open(img_path).convert("RGB") as img:
            img_t = self.transform(img)

        # Process all sequences (1 for train, 5 for val)
        processed_seqs = []
        for seq in seqs:
            seq_t = torch.tensor(seq, dtype=torch.long)
            if self.vocab is not None:
                vocab_size = len(self.vocab)
                unk_idx = self.vocab.get("<UNK>", vocab_size - 1)
                seq_t = torch.where(seq_t >= vocab_size, torch.tensor(unk_idx, dtype=torch.long), seq_t)
                seq_t = torch.where(seq_t < 0, torch.tensor(unk_idx, dtype=torch.long), seq_t)
            processed_seqs.append(seq_t)

        if not self.is_val:
            # Return a single tensor for training
            return img_t, processed_seqs[0], img_name
        else:
            # Return a list of tensors for validation
            return img_t, processed_seqs, img_name


def collate_fn(batch):
    """
    Custom collate_fn to handle both training (single caption) and validation (multiple captions).
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    img_names = [b[2] for b in batch]

    # Check if the batch is for validation (second element is a list of tensors)
    is_val_batch = isinstance(batch[0][1], list)

    if is_val_batch:
        # For validation, seqs are a list of lists of tensors. Don't stack them.
        seqs = [b[1] for b in batch]
    else:
        # For training, stack the sequence tensors
        seqs = torch.stack([b[1] for b in batch], dim=0)

    return images, seqs, img_names