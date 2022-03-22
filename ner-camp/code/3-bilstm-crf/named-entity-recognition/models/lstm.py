import torch.nn as nn
from torch.nn import LayerNorm
from models.modules import CRF


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 tag_dict, drop_p=0.1):
        super(LSTM, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size // 2,
                              batch_first=True, num_layers=2, dropout=drop_p,
                              bidirectional=True)
        self.dropout = nn.Dropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size)
        self.crf = CRF(in_features=hidden_size, num_tags=len(tag_dict))

    def _build_features(self, input_ids, input_mask):
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        sequence_output, _ = self.bilstm(embs)
        sequence_output = self.layer_norm(sequence_output)
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