# encoding=utf-8
import torch
from torch import nn
from config import *
from embedding import *
from transformer_layer import TransformerLayer
from LayerNorm import clones, Classify


class TransformerXl(nn.Module):
    def __init__(self, mul_attention_heads, hidden_size, mem_len, sentence_len, batch_size, dropout, vocab_size, eps, intermediate_size, num_layers, classify, pad_idx):
        super(TransformerXl, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.sentence_len = sentence_len
        self.mem_len = mem_len
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size, dropout)
        self.rel_pos_embedding = RelPositionEmbedding(hidden_size)
        self.transformer_layer = TransformerLayer(mul_attention_heads, hidden_size, mem_len, sentence_len, batch_size, dropout, vocab_size, eps, intermediate_size)
        self.transformer_layers = clones(self.transformer_layer, num_layers)
        self.init_mem = torch.randn(num_layers, batch_size, mem_len, hidden_size)
        self.classify = Classify(hidden_size, classify)

    def get_src_mask(self, seq, pad_idx):
        src_mask = (seq != pad_idx).unsqueeze(1)
        return src_mask.int()

    def get_trg_mask(self, trg, pad_idx):
        batch, trg_len = trg.size()
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        trg_mask = trg_mask & (1 - torch.triu(torch.ones(1, trg_len, trg_len), diagonal=1))
        return trg_mask

    def forward(self, input_segments, type_segments=None):
        batch_size, segments_count, sen_len = input_segments.size()
        mem = torch.zeros(self.num_layers, batch_size, segments_count*sen_len, self.hidden_size)
        memories = torch.cat([self.init_mem, mem], dim=2)
        for segment_num in range(segments_count):
            input_tokens = input_segments[:, segment_num, :]  # (batch, sen_len)
            # type_tokens = type_segments[:, segment_num, :]
            attention_mask = self.get_src_mask(input_tokens, self.pad_idx)
            token_embedding = self.token_embedding(input_tokens)
            rel_pos_embedding = self.rel_pos_embedding(self.sentence_len, self.mem_len)
            for layer_num in range(self.num_layers):
                if layer_num == 0:
                    rel_pos_attention, new_memories = self.transformer_layers[layer_num](token_embedding, rel_pos_embedding, memories, segment_num, layer_num, attention_mask)
                    memories = new_memories
                else:
                    rel_pos_attention, new_memories = self.transformer_layers[layer_num](rel_pos_attention, rel_pos_embedding, memories, segment_num, layer_num, attention_mask)
                    memories = new_memories
        return self.classify(rel_pos_attention)


if __name__ == '__main__':
    segment_count = 1
    input_segments = torch.arange(128).reshape(args.batch_size, args.sentence_len).expand(args.batch_size, segment_count, args.sentence_len)
    print('999', input_segments.size())
    transformer_xl = TransformerXl(args.mul_attention_heads, args.hidden_size, args.mem_len, args.sentence_len, args.batch_size, args.dropout, args.vocab_size, args.eps, args.intermediate_size, args.num_layers, args.classify, args.pad_idx)
    class_res = transformer_xl(input_segments)
    print(class_res.size())












