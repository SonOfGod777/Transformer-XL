import torch
from torch import nn
from config import *
from LayerNorm import *


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super(TokenEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.layer_norm = LayerNorm(args.eps, args.hidden_size)

    def forward(self, input_tokens):  # (batch, sen)
        token_embedding = self.token_embedding(input_tokens)
        token_embedding = self.layer_norm(token_embedding)
        token_embedding = self.dropout(token_embedding)
        return token_embedding


class RelPositionEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(RelPositionEmbedding, self).__init__()
        self.inv_freq = 1 / 10000 ** (torch.arange(0, hidden_size, 2) / hidden_size)

    def forward(self, seq_len, mem_len):
        pos_seq = torch.arange(1, seq_len+mem_len+1)
        sin_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
        return pos_emb


if __name__ == '__main__':
    input_token = torch.arange(8).reshape(2, 4)
    token_embed = TokenEmbedding(args.vocab_size, args.hidden_size, args.dropout)
    pp = token_embed(input_token)
    print(pp)
#
#     pos = RelPositionEmbedding(args.hidden_size)
#     print(pos(args.sentence_len, args.mem_len))

