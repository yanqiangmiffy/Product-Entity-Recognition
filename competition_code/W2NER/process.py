import json
import warnings

import jieba
import pandas as pd
from ark_nlp.factory.utils.conlleval import get_entity_bio
from tqdm import tqdm

warnings.filterwarnings("ignore")

print([w for w in jieba.cut('OPPO闪充充电器 X9070 X9077 R5 快充头通用手机数据线 套餐【2.4充电头+数据线 】 安卓 1.5m')])
result = jieba.tokenize(u'OPPO闪充充电器 X9070 X9077 R5 快充头通用手机数据线 套餐【2.4充电头+数据线 】 安卓 1.5m')
print([w for w in result])


def get_data(bio_file='data/gaiic/train_data/train.txt'):
    datalist = []
    with open(bio_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')

        text = []
        labels = []
        label_set = set()

        for line in lines:
            if line == '\n':
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx + 1]
                    })

                if text == '':
                    continue

                datalist.append({
                    'text': text,
                    'label': entity_labels
                })

                text = []
                labels = []

            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
    return datalist

from sklearn.model_selection import train_test_split
datalist = get_data('data/gaiic/train_data/train.txt')


data_train,data_dev=train_test_split(datalist,test_size=0.1,random_state=42)
train_data_df = pd.DataFrame(data_train)
dev_data_df = pd.DataFrame(data_dev)

train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))


datalist = get_data('data/gaiic/preliminary_test_a/nezha.txt')
test_data_df = pd.DataFrame(datalist)
test_data_df['label'] = test_data_df['label'].apply(lambda x: str(x))


def save_ner_json(df, save_path='data/gaiic/train.json'):
    train_data = []

    for index, row in tqdm(df.iterrows()):
        sentence = [word for word in row['text']]
        ner = []
        for entity_label in eval(row['label']):
            # print(entity_label)
            entity = {
                'index': [i for i in range(entity_label['start_idx'], entity_label['end_idx'] + 1)],
                'type': entity_label['type']
            }
            ner.append(entity)
        word = []
        words = jieba.tokenize(row['text'])
        for word_index in words:
            # ('OPPO', 0, 4),
            word.append([index for index in range(word_index[1], word_index[2])])
        train_data.append({'sentence': sentence, 'ner': ner, 'word': word})

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    save_ner_json(train_data_df, 'data/gaiic/train.json')
    save_ner_json(dev_data_df, 'data/gaiic/dev.json')
    save_ner_json(test_data_df, 'data/gaiic/test.json')
