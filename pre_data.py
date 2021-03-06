# encoding=utf-8
from token_model.tokenization import Tokenizer
from XL_Model.config import *
import torch


class BuildData(object):
    def __init__(self, CLS=['[CLS]']):
        self.cls = CLS
        self.Tokenizer = Tokenizer(args.vocab_path)
        self.sentence_len = args.sentence_len

    def load_data(self, path):
        output = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                text, label = line.split('\t')
                tokens = self.Tokenizer.tokenize(text)
                tokens = self.cls + tokens
                text_ids = self.Tokenizer.convert_tokens_to_ids(tokens)
                sentence_count = 0
                texts_ids = []
                while sentence_count * self.sentence_len < len(text_ids):
                    curr_ids = text_ids[sentence_count*self.sentence_len:min((sentence_count+1)*self.sentence_len, len(text_ids))]
                    if len(curr_ids) < self.sentence_len:
                        curr_ids += (self.sentence_len-len(curr_ids)) * [0]
                    texts_ids.append(curr_ids)
                    sentence_count += 1
                output.append((texts_ids, int(str(label).strip())))
        return output

    def build_data(self):
        train_data = self.load_data(args.train_path)
        dev_data = self.load_data(args.dev_path)
        test_data = self.load_data(args.test_path)
        return train_data, dev_data, test_data


class BatchData(object):
    def __init__(self, data, index=0, batch_size=args.batch_size):
        self.index = index
        self.device = args.device
        self.batch_size = batch_size
        self.data = data
        self.batch_nums = len(self.data) // self.batch_size
        self.residue = False
        if len(self.data) % self.batch_size != 0:
            self.residue = True

    def to_tensor(self, batch):
        x = torch.LongTensor([_[0] for _ in batch]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batch]).to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.batch_nums:
            batch = self.data[self.index*self.batch_size:len(self.data)]
            self.index += 1
            return self.to_tensor(batch)
        elif self.index >= self.batch_nums:
            self.index = 0
            raise StopIteration
        else:
            batch = self.data[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            return self.to_tensor(batch)

    def __iter__(self):
        return self


if __name__ == '__main__':
    # path = args.train_path
    pp = BuildData()
    train, _, _ = pp.build_data()
    train_pp = BatchData(train)
    for k in train_pp:
        print('k', k[0])
        print('v', k[1])








