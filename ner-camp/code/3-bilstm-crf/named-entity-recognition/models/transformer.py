import torch.nn as nn
from models.utils import get_attn_pad_mask
from models.modules import CRF, TransformerEncoder, TokenEmbedding, PositionalEncoding


class Transformer(nn.Module):

    def __init__(self, vocab_size, tag_dict, num_blocks, model_dim, num_heads, feedforward_dim, drop_p=0.1):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = TokenEmbedding(vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.dropout = nn.Dropout(drop_p)
        self.crf = CRF(in_features=model_dim, num_tags=len(tag_dict))
        self.encoder = TransformerEncoder(num_blocks, model_dim, num_heads, feedforward_dim, drop_p)

    def _build_features(self, input_ids, input_mask):
        embs = self.embedding(input_ids)
        embs = self.pos_encoding(embs)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        attn_mask = get_attn_pad_mask(input_ids, input_ids)
        sequence_output = self.encoder(embs, attn_mask)
        return sequence_output

    def forward(self, input_ids, input_mask):
        features = self._build_features(input_ids, input_mask)
        scores, tags = self.crf(features, input_mask)
        return scores, tags

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self._build_features(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.loss(features, input_tags, input_mask)
        else:
            return features