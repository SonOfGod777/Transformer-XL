import torch
from torch import nn
from LayerNorm import LayerNorm
from RelPosAttention import *
from config import *
from embedding import *


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention_x):
        dense_x = self.dense1(attention_x)
        dense_x = torch.relu(dense_x)
        dense_x = self.dense2(dense_x)
        return self.dropout(dense_x)


class TransformerLayer(nn.Module):
    def __init__(self, mul_attention_heads, hidden_size, mem_len, sentence_len, batch_size, dropout, vocab_size, eps, intermediate_size):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(eps, hidden_size)
        self.relposAttention = RelPosMulSelfAttention(mul_attention_heads, hidden_size, mem_len, sentence_len, batch_size, dropout)
        self.dense = FeedForward(hidden_size, intermediate_size, dropout)

    def forward(self, token_embedding, rel_pos_embedding, mems, segment_num, layer_num, attention_mask=None):
        attention_x, mems = self.relposAttention(token_embedding, rel_pos_embedding, mems, segment_num, layer_num, attention_mask)
        attention_x += token_embedding
        attention_x = self.layer_norm(attention_x)

        dense_x = self.dense(attention_x)
        dense_x += attention_x
        dense_x = self.layer_norm(dense_x)
        return dense_x, mems


# if __name__ == '__main__':
#     segment_count = 2
#     init_mem = torch.randn(args.num_layers, args.batch_size, args.mem_len, args.hidden_size)
#     mem = torch.zeros([args.num_layers, args.batch_size, segment_count * args.sentence_len, args.hidden_size])
#     mems = torch.cat([init_mem, mem], dim=2)
#     print('mem', mems.size())
#
#     input_token = torch.arange(128).reshape(args.batch_size, args.sentence_len)
#     token_embed = TokenEmbedding(args.vocab_size, args.hidden_size, args.dropout)
#     token_embedding = token_embed(input_token)
#
#     pos = RelPositionEmbedding(args.hidden_size)
#     rel_pos_embedding = pos(args.sentence_len, args.mem_len)
#
#     attention_mask = get_src_mask(input_token, 0)
#
#     transformer_layer = TransformerLayer(args.mul_attention_heads, args.hidden_size, args.mem_len, args.sentence_len, args.batch_size, args.dropout, args.vocab_size, args.eps, args.intermediate_size)
#     rel_pos_attention, mems = transformer_layer(token_embedding, rel_pos_embedding, mems, 1, 1, attention_mask)
#     print('rel_pos_attention', rel_pos_attention.size())
#     print('mems', mems.size())
