import random
import torch
from utils import to_index


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle, word_dict, tag_dict, seed, sort=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.sort = sort
        self.examples = None
        self.features = None
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.reset()

    def reset(self):
        self.examples = self.preprocess(self.dataset)
        if self.sort:
            self.examples = sorted(self.examples, key=lambda x: x[2], reverse=True)
        if self.shuffle:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        self.features = [self.examples[i:i + self.batch_size] for i in range(0, len(self.examples), self.batch_size)]
        print(f"{len(self.features)} batches created")

    def preprocess(self, dataset):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in dataset:
            text = d['context']
            tokens = [to_index(self.word_dict, w) for w in text.split(" ")]
            x_len = len(tokens)
            text_tag = d['tag']
            tag_ids = [self.tag_dict[tag] for tag in text_tag.split(" ")]
            processed.append((tokens, tag_ids, x_len, text, text_tag))
        return processed

    def get_long_tensor(self, tokens_list, batch_size, mask=False):
        """ Convert list of list of tokens to a padded LongTensor. """
        token_len = max(len(x) for x in tokens_list)
        tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_ = torch.LongTensor(batch_size, token_len).fill_(0)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
            if mask:
                mask_[i, :len(s)] = torch.tensor([1] * len(s), dtype=torch.long)
        if mask:
            return tokens, mask_
        return tokens

    def sort_all(self, batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """
        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """ Get a batch with index. """
        if not isinstance(index, int):
            raise TypeError
        if index < 0 or index >= len(self.features):
            raise IndexError
        batch = self.features[index]
        batch_size = len(batch)
        batch = list(zip(*batch))
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = self.sort_all(batch, lens)
        chars = batch[0]
        input_ids, input_mask = self.get_long_tensor(chars, batch_size, mask=True)
        label_ids = self.get_long_tensor(batch[1], batch_size)
        input_lens = [len(x) for x in batch[0]]
        return input_ids, input_mask, label_ids, input_lens
