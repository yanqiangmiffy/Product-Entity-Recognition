import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import log_sum_exp, scaled_dot_product


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """ 独立的 CRF 模块，已经包括了 START 和 STOP标签，可以直接用于 LSTM 或 Transformer 的输出 """
    def __init__(self, in_features, num_tags):
        """ Construction

        :param in_features: input dimension
        :param num_tag: number of named entity tags
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # 非标准化转移概率，其中T[i][j]表示由状态j转移到状态i的非标准化概率
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """ 根据 LSTM 或 Transformer 的输出特征得到最优序列输出

        :param features: [batch_size, seq_len, d_model]
        :param masks: [batch_size, seq_len] masks
        :return: (best_score, best_paths)
            best_score: [batch_size]
            best_paths: [batch_size, seq_len]
        """
        features = self.fc(features)
        return self._viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """ CRF 模块损失

        :param features: [batch_size, seq_len, d_model]
        :param ys: tags, [batch_size, seq_len]
        :param masks: masks for padding, [batch_size, seq_len]
        :return: loss
        """
        features = self.fc(features)

        seq_len = features.size(1)
        masks_ = masks[:, :seq_len].float()

        forward_score = self._forward_algorithm(features, masks_)
        gold_score = self._score_sentence(features, ys[:, :seq_len].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def _score_sentence(self, features, tags, masks):
        """ Golden 序列的非标准化概率

        :param features: [batch_size, seq_len, n_tags]
        :param tags: [batch_size, seq_len]
        :param masks: batch_size, seq_len]
        :return: [batch_size] score in the log space
        """
        batch_size, seq_len, n_tags = features.shape

        # emission 非标准化概率
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

        # transition 非标准化概率
        start_tag = torch.full((batch_size, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [batch_size, seq_len+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # STOP 非标准化转移概率
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [batch_size]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def _viterbi_decode(self, features, masks):
        """ Viterbi 算法解码得到最优序列

        :param features: [batch_size, seq_len, n_tags]
        :param masks: [batch_size, seq_len]
        :return: (best_score, best_paths)
            best_score: [batch_size]
            best_paths: [batch_size, seq_len]
        """
        batch_size, seq_len, n_tags = features.shape

        bps = torch.zeros(batch_size, seq_len, n_tags, dtype=torch.long, device=features.device)  # back pointers

        # 在对数空间计算非标准化概率，表示当前时序到达每一个状态的得分
        max_score = torch.full((batch_size, n_tags), IMPOSSIBLE, device=features.device)  # [batch_size, n_tags]
        max_score[:, self.start_idx] = 0

        for t in range(seq_len):
            mask_t = masks[:, t].unsqueeze(1)  # [batch_size, 1]
            emit_score_t = features[:, t]  # [batch_size, n_tags]

            # [batch_size, 1, n_tags] + [n_tags, n_tags]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [batch_size, n_tags, n_tags]
                                                                     # 这里的score[i][j]表示由tag_j转移到tag_i的得分
                                                                     # 需要记录tag_i是由哪一个tag_j转移得到
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # STOP
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # 根据 back pointers 指针解码恢复最优序列路径
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(batch_size):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # 序列起始的标签来自虚拟 START，在恢复的路径中删掉再翻转时序
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def _forward_algorithm(self, features, masks):
        """ 前向算法计算 CRF 中的配分函数，即所有路径的非标准化概率和
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])

        :param features: features. [batch_size, seq_len, n_tags]
        :param masks: [batch_size, seq_len] masks
        :return: [batch_size], score in the log space
        """
        batch_size, seq_len, n_tags = features.shape

        scores = torch.full((batch_size, n_tags), IMPOSSIBLE, device=features.device)  # [batch_size, n_tags]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, n_tags, n_tags]

        # 按照时序迭代
        for t in range(seq_len):
            emit_score_t = features[:, t].unsqueeze(2)  # [batch_size, n_tags, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [batch_size, 1, n_tags] +
                                                                  # [1, n_tags, n_tags] +
                                                                  # [batch_size, n_tags, 1] =>
                                                                  # [batch_size, n_tags, n_tags]
            score_t = log_sum_exp(score_t)  # [batch_size, n_tags]

            mask_t = masks[:, t].unsqueeze(1)  # [batch_size, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores


class MultiheadAttention(nn.Module):
    """ 多头注意力机制 """
    def __init__(self, input_dim, model_dim, num_heads):
        """ Construction

        :param input_dim: input dimension
        :param model_dim: model dimension
        :param num_heads: number of attention heads
        """
        super(MultiheadAttention, self).__init__()
        assert model_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Query Key Value 仿射矩阵
        self.qkv_proj = nn.Linear(input_dim, 3 * model_dim)
        self.o_proj = nn.Linear(model_dim, model_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # xavier 初始化权重参数
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """ 多头注意力计算

        :param x: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :param return_attention: default false
        :return: [batch_size, seq_len, d_model], attention output
        """
        batch_size, seq_len, model_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, d_head]
        q, k, v = qkv.chunk(3, dim=-1)

        # 使用 mask 遮蔽补零位置处的注意力得分
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [batch_size, seq_len, n_heads, d_head]
        values = values.reshape(batch_size, seq_len, model_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    """ Transformer Encoder Block """
    def __init__(self, input_dim, num_heads, feedforward_dim, drop_p=0.0):
        """ Contruction

        :param input_dim: input dimension
        :param num_heads: number of attention heads
        :param feedforward_dim: feed-forward dimension
        :param drop_p: dropout probability
        """
        super(EncoderBlock, self).__init__()

        # multi-head attention sublayer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # position-wise feed-forward sublayer
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.Dropout(drop_p),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dim, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, mask=None):
        # multi-head attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # position-wise feed-forward part
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, num_blocks, input_dim, num_heads, feedforward_dim, drop_p=0.0):
        """ Construction

        :param num_blocks: number of encoder block
        :param input_dim: input dimension (model dimension)
        :param num_heads: number of attention heads
        :param feedforward_dim: feed-forward dimension
        :param drop_p: dropout probability
        """
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(input_dim, num_heads, feedforward_dim, drop_p) for _ in range(num_blocks)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        """ 输入数据对应的注意力激活特征

        :param x: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :return: attention maps
        """
        attention_maps = []
        for block in self.blocks:
            _, attn_map = block.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = block(x)
        return attention_maps


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, model_dim):
        """ Construction

        :param vocab_size: vocab size
        :param model_dim: model (embedding) dimension
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        # 和 Transformer 论文中保持一致
        return self.embedding(x) * math.sqrt(self.model_dim)
        # return self.embedding(x)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_len=5000):
        """ Construction

        :param model_dim: model (embedding) dimension
        :param max_len: maximum length
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer 表示参数属于模型的一部分但不是 Parameter 类型
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
