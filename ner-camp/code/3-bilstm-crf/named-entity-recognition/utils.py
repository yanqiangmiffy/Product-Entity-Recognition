import os
import json
import torch
from collections import Counter


PAD = '[PAD]'
UNK = '[UNK]'
files = ['train.json', 'dev.json', 'test.json']


def build_vocab(data_dir, min_freq=None, max_freq=None):
    word_dict = dict()
    ctrl_symbols = [PAD, UNK]
    for idx, sym in enumerate(ctrl_symbols):
        word_dict[sym] = idx

    word_counter = Counter()
    for file in files:
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                text = line['text']
                word_counter.update(list(text))

    max_size = min(max_freq, len(word_counter)) if max_freq else None
    words = word_counter.most_common(max_size)
    if min_freq is not None:
        words = filter(lambda kv: kv[1] >= min_freq, words)
    offset = len(word_dict)
    word_dict.update({w: i + offset for i, (w, _) in enumerate(words)})
    return word_dict


def create_examples(input_path, mode='train'):
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            json_d = {}
            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label', None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index: end_index + 1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1: end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            json_d['id'] = f"{mode}_{idx}"
            json_d['context'] = " ".join(words)
            json_d['tag'] = " ".join(labels)
            json_d['raw_context'] = "".join(words)
            idx += 1
            examples.append(json_d)
    return examples


def get_train_examples(data_dir):
    return create_examples(os.path.join(data_dir, "train.json"), "train")


def get_dev_examples(data_dir):
    return create_examples(os.path.join(data_dir, "dev.json"), "dev")


def get_test_examples(data_dir):
    return create_examples(os.path.join(data_dir, "test.json"), "test")


def to_index(vocab, word):
    if word in vocab:
        index = vocab[word]
    elif UNK in vocab:
        index = vocab[UNK]
    else:
        raise ValueError("word {} not in vocabulary".format(word))
    return index


def get_entities(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def save(model, model_path):
    torch.save(model.state_dict(), model_path)