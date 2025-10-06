import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import heapq

class EncoderCNN(nn.Module):
  """
  ResNet50 backbone that returns spatial features and a pooled global vector.
  features: (B, N, feat_dim), global_feat: (B, feat_embed_dim)
  """
  def __init__(self, feat_dim=2048, feat_embed_dim=512, train_backbone=False):
    super().__init__()
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-2]  # remove avgpool & fc
    self.backbone = nn.Sequential(*modules)
    for p in self.backbone.parameters():
        p.requires_grad = train_backbone
    self.feat_dim = feat_dim
    self.feat_embed = nn.Linear(feat_dim, feat_embed_dim)

  def forward(self, images):
    x = self.backbone(images)
    B, C, H, W = x.size()
    x = x.view(B, C, H*W).permute(0, 2, 1)
    # project high-dim features to feat_embed_dim
    feats = self.feat_embed(x)
    # global feature for init: mean over regions
    global_feat = feats.mean(dim=1)
    return feats, global_feat

class Attention(nn.Module):
  def __init__(self, encoder_dim, hidden_dim, attn_dim):
    """
    encoder_dim: The dimension of the encoded image features (feat_embed_dim)
    hidden_dim: The dimension of the decoder's hidden state
    attn_dim: The dimension of the attention network
    """
    super().__init__()
    self.W_feat = nn.Linear(encoder_dim, attn_dim) 
    self.W_hidden = nn.Linear(hidden_dim, attn_dim)
    self.v = nn.Linear(attn_dim, 1)

  def forward(self, feats, hidden):
    f_proj = self.W_feat(feats)
    # hidden has shape (B, hidden_dim), h_proj will have (B, attn_dim)
    # unsqueeze adds a dimension for broadcasting: (B, 1, attn_dim)
    h_proj = self.W_hidden(hidden).unsqueeze(1)
    e = torch.tanh(f_proj + h_proj)
    scores = self.v(e).squeeze(2)
    weights = F.softmax(scores, dim=1)
    context = (feats * weights.unsqueeze(2)).sum(dim=1)
    return context, weights

class DecoderWithAttention(nn.Module):
  def __init__(self, embed_dim, hidden_dim, vocab_size, feat_dim=2048,
              feat_embed_dim=512, attn_dim=512, num_layers=1, dropout=0.5):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    self.attention = Attention(feat_embed_dim, hidden_dim, attn_dim)
    self.lstm = nn.LSTM(embed_dim + feat_embed_dim, hidden_dim,
                        num_layers=num_layers, batch_first=True,
                        dropout=dropout if num_layers>1 else 0.0)
    self.init_fc = nn.Linear(feat_embed_dim, hidden_dim)
    self.init_cell_fc = nn.Linear(feat_embed_dim, hidden_dim)
    self.fc_out = nn.Linear(hidden_dim, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, feats, captions):
    """
    feats: (B, N, feat_embed_dim)
    captions: (B, L) full sequences with <START> and <END>
    return: logits (B, L-1, vocab)
    """
    B, N, Fd = feats.size()
    emb = self.embed(captions[:, :-1])
    emb = self.dropout(emb)

    global_feat = feats.mean(dim=1)
    h = self.init_fc(global_feat).unsqueeze(0)
    c = self.init_cell_fc(global_feat).unsqueeze(0)

    outs = []
    hidden = (h, c)
    for t in range(emb.size(1)):
      emb_t = emb[:, t, :]
      h_top = hidden[0][-1]
      context, _ = self.attention(feats, h_top)
      lstm_input = torch.cat([emb_t, context], dim=1).unsqueeze(1)
      out, hidden = self.lstm(lstm_input, hidden)
      out = out.squeeze(1)
      out = self.dropout(out)
      logit = self.fc_out(out)
      outs.append(logit.unsqueeze(1))
    logits = torch.cat(outs, dim=1)
    return logits

  def beam_search_decode(self, feats, start_token, end_token, max_len=30, beam_width=3):
    """
    Performs beam search decoding for a batch of images.
    """
    B, N, Fd = feats.size()
    global_feat = feats.mean(dim=1)
    
    batch_predictions = []

    for i in range(B):
      single_feats = feats[i].unsqueeze(0)
      h = self.init_fc(global_feat[i]).unsqueeze(0).unsqueeze(0)
      c = self.init_cell_fc(global_feat[i]).unsqueeze(0).unsqueeze(0)
      hidden = (h, c)

      beam = [(-0.0, [start_token], hidden)]
      completed_sequences = []

      for _ in range(max_len):
        new_beam = []
        for log_prob, seq, hidden_state in beam:
          if seq[-1] == end_token or len(completed_sequences) >= beam_width:
            completed_sequences.append((log_prob, seq))
            continue

          input_tok = torch.tensor([seq[-1]], device=feats.device).unsqueeze(0)
          emb = self.embed(input_tok)
          
          h_top = hidden_state[0][-1]
          
          context, _ = self.attention(single_feats, h_top)
          
          lstm_input = torch.cat([emb.squeeze(1), context], dim=1).unsqueeze(1)
          out, new_hidden = self.lstm(lstm_input, hidden_state)
          
          logit = self.fc_out(out.squeeze(1))
          log_probs = F.log_softmax(logit, dim=-1)
          
          top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
          
          for k in range(beam_width):
            new_seq = seq + [top_indices[0, k].item()]
            new_log_prob = log_prob + top_log_probs[0, k].item()
            new_beam.append((new_log_prob, new_seq, new_hidden))

        beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]
        if not beam:
          break
      
      completed_sequences.extend([(log_prob, seq) for log_prob, seq, _ in beam])

      if completed_sequences:
        best_seq = sorted(completed_sequences, key=lambda x: x[0], reverse=True)[0][1]
        batch_predictions.append(best_seq)
      else:
        batch_predictions.append([start_token, end_token])
  
    return batch_predictions