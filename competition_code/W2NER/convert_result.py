import json

with open('output.json', 'r', encoding='utf-8') as f:
    preds = json.load(f)
print(len(preds))

with open('w2ner.txt', 'w', encoding='utf-8') as f:
    for example in preds:
        text = example['sentence']
        tags = ['O'] * len(text)
        for entity_res in example['entity']:
            # {'text': ['充', '电', '器'], 'index': [6, 7, 8], 'type': '4'}
            for i in range(len(entity_res['index'])):
                if i==0:
                    tags[entity_res['index'][i]]='B-'+entity_res['type']
                else:
                    tags[entity_res['index'][i]]='I-'+entity_res['type']

        for char, tag in zip(text, tags):
            f.write('{} {}'.format(char, tag))
            f.write('\n')
        f.write('\n')
