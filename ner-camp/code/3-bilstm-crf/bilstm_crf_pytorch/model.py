from torch.nn import LayerNorm
import torch.nn as nn
from crf import CRF


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class NERModel(nn.Module):
    """
    Bilstm+CRF
    """

    def __init__(self, vocab_size, embedding_size, hidden_size,
                 label2id, device, drop_p=0.1):
        super(NERModel, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )  # Embedding层

        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                              batch_first=True, num_layers=2, dropout=drop_p,
                              bidirectional=True)  # BiLSTM层

        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = LayerNorm(hidden_size * 2)

        self.classifier = nn.Linear(hidden_size * 2, len(label2id))  # 全连接层 获取发射分数
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)  # CRF层

    def forward(self, inputs_ids, input_mask):
        """
        输入文本id序列 获取发射分数
        :param inputs_ids:
        :param input_mask:
        :return:
        """
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)  # dropout 避免过拟合
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)
        features = self.classifier(seqence_output) # Linear
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features

# sample 1：长度 4 PAD PAD：O
# sample 2：长度 10
# max：10
