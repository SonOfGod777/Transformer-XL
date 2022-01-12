import torch
from torch import nn
from config import *
from LayerNorm import clones
from embedding import *


class RelPosMulSelfAttention(nn.Module):
    def __init__(self, mul_attention_heads, hidden_size, mem_len, sentence_len, batch_size, dropout):
        super(RelPosMulSelfAttention, self).__init__()
        self.mul_attention_heads = mul_attention_heads
        self.hidden_size = hidden_size
        self.mem_len = mem_len
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.dropout = nn.Dropout(dropout)
        assert self.hidden_size % self.mul_attention_heads == 0
        self.attention_head_size = self.hidden_size // self.mul_attention_heads
        self.linears = clones(nn.Linear(self.hidden_size, self.hidden_size), 5)
        self.u = torch.randn(self.mul_attention_heads, self.attention_head_size)
        self.v = torch.randn(self.mul_attention_heads, self.attention_head_size)

    def rel_shift(self, bd):
        q_len, k_len, batch_size, mul_attention_heads = bd.size()
        for qi in range(q_len):
            bd[qi] = torch.cat([bd[qi][q_len-1-qi:], torch.zeros(q_len-1-qi, batch_size, mul_attention_heads)], dim=0)
        return bd

    def update_mem(self, last_hidden, memories, layer_num, segment_num):
        with torch.no_grad():
            for batch in range(len(memories[layer_num])):
                memories[layer_num][batch][self.mem_len+segment_num*self.sentence_len:self.mem_len+(segment_num+1)*self.sentence_len] = last_hidden[batch]
        return memories

    def forward(self, token_embedding, rel_pos_embedding, mems, segment_num, layer_num, attention_mask=None):
        mem = mems[layer_num][:, segment_num*self.mem_len:(segment_num+1)*self.mem_len, :]
        q_x = token_embedding
        k_x = torch.cat([token_embedding, mem], dim=1)
        k_len = k_x.size()[1]
        k_r = rel_pos_embedding
        v_x = torch.cat([token_embedding, mem], dim=1)
        q_x, k_x, v_x = [linear(x).view(-1, self.batch_size, self.mul_attention_heads, self.attention_head_size) for linear, x in zip(self.linears, [q_x, k_x, v_x])]
        k_r = self.linears[4](k_r).view(-1, self.mul_attention_heads, self.attention_head_size)

        qx_u = q_x + self.u
        ac = torch.einsum('ibnd,jbnd->ijbn', [qx_u, k_x])
        qx_v = q_x + self.v
        bd = torch.einsum('ibnd,jnd->ijbn', [qx_v, k_r])
        bd = self.rel_shift(bd)

        rel_pos_attention = ac + bd
        rel_pos_attention = rel_pos_attention / torch.sqrt(torch.tensor(float(self.attention_head_size)))

        if attention_mask is not None:  # (batch, 1, sen_len)
            src_mem = torch.ones(self.batch_size, 1, self.mem_len)
            src_mask = torch.cat([src_mem, attention_mask], dim=-1).unsqueeze(1)
            src_mask = (1 - src_mask) * 1e9
            rel_pos_attention = rel_pos_attention.contiguous().view(self.batch_size, self.mul_attention_heads, self.sentence_len, k_len)
            rel_pos_attention -= src_mask
            rel_pos_attention = rel_pos_attention.contiguous().view(self.sentence_len, k_len, self.batch_size, self.mul_attention_heads)
        rel_pos_attention = nn.Softmax(dim=-1)(rel_pos_attention)
        rel_pos_attention = self.dropout(rel_pos_attention)
        rel_pos_attention = torch.einsum('ijbn,jbnd->ibnd', [rel_pos_attention, v_x])
        rel_pos_attention = rel_pos_attention.contiguous().view(self.batch_size, self.sentence_len, self.hidden_size)
        rel_pos_attention = self.linears[-1](rel_pos_attention)

        mems = self.update_mem(rel_pos_attention, mems, layer_num, segment_num)
        return rel_pos_attention, mems


# if __name__ == '__main__':
#     segment_count = 2
#     init_mem = torch.randn(args.num_layers, args.batch_size, args.mem_len, args.hidden_size)
#     mem = torch.zeros([args.num_layers, args.batch_size, segment_count * args.sentence_len, args.hidden_size])
#     mems = torch.cat([init_mem, mem], dim=2)
#
#     input_token = torch.arange(128).reshape(args.batch_size, args.sentence_len)
#     token_embed = TokenEmbedding(args.vocab_size, args.hidden_size, args.dropout)
#     token_embedding = token_embed(input_token)
#
#     pos = RelPositionEmbedding(args.hidden_size)
#     rel_pos_embedding = pos(args.sentence_len, args.mem_len)
#
#     attention_mask = get_src_mask(input_token, 0)
#     print(attention_mask.size())
#
#     pp = RelPosMulSelfAttention(args.mul_attention_heads, args.hidden_size, args.mem_len, args.sentence_len, args.batch_size, args.dropout)
#     pp(token_embedding, rel_pos_embedding, mems, 1, 1, attention_mask)