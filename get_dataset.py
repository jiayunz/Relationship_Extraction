# encoding: utf-8
import os
from tqdm import tqdm
import re
import csv
import thulac
import json
import numpy as np
import math

all_relationships = [
    '母子',
    '母女',
    '父女',
    '父子',
    #'父', # -> 父子
    '兄弟',
    '姐妹',
    '兄妹',
    '姐弟',
    '叔侄',
    '舅甥',
    '干母子',
    '干父子',
    '干姐弟',
    '干兄弟',
    '干父女',
    '翁媳',
    '翁婿',
    '同学',
    '搭档',
    #'拍档', # -> 拍档
    # 'relation',
    '好友',
    # '密友', # -> 好友
    '夫妻',
    # '准夫妻', # -> 夫妻
    '情侣',
    #'恋人', # -> 情侣
    # '情人', # -> 情侣
    '前任',
    #'前任夫妻', # -> 前任
    #'旧爱', # -> 前任
    '绯闻',
    #'绯闻男友', # -> 绯闻
    '师徒',
    '同门',
    '组合',
    '组员'
]
MAX_SEQLEN = 3

def get_sentence_labels():
    category_dir = "/bdata/jiayunz/THUCNews/娱乐/"
    files = os.listdir(category_dir)
    pseg = thulac.thulac()
    labels = {}
    # 读取已知关系标注
    with open('ylq_star_relations.csv', 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            labels[(row[0], row[3])] = row[2]

    print(labels)

    with open('labeled_data.txt', 'w') as wf:
        for f in tqdm(files, total=len(files)):
            with open(category_dir + f, 'r') as rf:
                title = rf.readline()
                content = rf.read()
                #sentences = SnowNLP(content).sentences
                sentences = re.split(r'[。！？]', content)
                # todo: jieba加入明星名字词典。
                for s in sentences:
                    people_in_sentence = []
                    poss = pseg.cut(s)

                    for w in poss:
                        if w[1] == 'np' and len(w[0]) > 1:
                            people_in_sentence.append(w[0])
                    people_in_sentence = list(set(people_in_sentence))

                    if len(people_in_sentence) > 1:
                        for i, u in enumerate(people_in_sentence):
                            for j, v in enumerate(people_in_sentence[i+1:]):
                                if (u, v) in labels:
                                    wf.write(json.dumps({'sentence': s, 'label': (u, v, labels[(u, v)])}, ensure_ascii=False) + '\n')
                                elif (v, u) in labels:
                                    wf.write(json.dumps({'sentence': s, 'label': (u, v, labels[(v, u)])}, ensure_ascii=False) + '\n')

# 统计所有标签
def get_all_labels():
    relationships = {}
    labels = []
    with open('ylq_star_relations.csv', 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            relationships[(row[0], row[3])] = row[2]
            labels.append(row[2])

    labels = set(labels)
    print(relationships)
    #print(labels)

def make_dataset():
    pseg = thulac.thulac()
    with open('words_list.json', 'w') as wf:
        with open('labeled_data.txt', 'r') as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                content = json.loads(line)
                # words list
                words_list = pseg.cut(content['sentence'])

                if content['label'][2] == '父':
                    content['label'][2] = '父子'
                elif content['label'][2] == '拍档':
                    content['label'][2] = '搭档'
                elif content['label'][2] == '恋人' or content['label'][2] == '情人':
                    content['label'][2] = '情侣'
                elif content['label'][2] == '密友':
                    content['label'][2] = '好友'
                elif content['label'][2] == '准夫妻':
                    content['label'][2] = '夫妻'
                elif content['label'][2] == '前任夫妻' or content['label'][2] == '旧爱':
                    content['label'][2] = '前任'
                elif content['label'][2] == '绯闻男友':
                    content['label'][2] = '绯闻'

                wf.write(json.dumps({
                    "text": content['sentence'],
                    "words_list": words_list,
                    "people": content['label'][:2],
                    "relationship": content['label'][2]
                }, ensure_ascii=False) + '\n')

def delete_duplicate_sentences():
    text_existed = []
    with open('words_list.json') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            content = json.loads(line)

            text_existed.append(content['text'])


def get_relationships():
    relationships = {}
    labels = {}
    with open('words_list.json') as rf:
        for line in rf.readlines():
            content = json.loads(line)
            # # 同一对relationship整理到一起
            # (u, v)
            if tuple(content['people']) in relationships:
                r = tuple(content['people'])
            # (v, u)
            elif (content['people'][1], content['people'][0]) in relationships:
                r = (content['people'][1], content['people'][0])
            else:
                r = None

            new_words_list = []
            for j, w in enumerate(content['words_list']):
                if re.search('^[\n\s]$', w[0]) or w[1] in ['w', 'e', 'o', 'g', 'x', 'q', 'mq', 'm', 't', 'f']:
                    continue
                # 把名字换成target_person
                elif w[0] == content['people'][0]:
                    new_words_list.append('目标人物1')
                elif w[0] == content['people'][1]:
                    new_words_list.append('目标人物2')
                # 其他人名
                elif w[1] == 'np':
                    new_words_list.append('其他人物')
                else:
                    new_words_list.append(w[0])

            if r:
                relationships[r]["sentences"].append(' '.join(new_words_list))
            else:
                relationships[tuple(content['people'])] = {
                    "sentences": [' '.join(new_words_list)],
                    "people": content['people'],
                    "label": content["relationship"]
                }
                try:
                    labels[content["relationship"]] += 1
                except:
                    labels[content["relationship"]] = 1

    print(labels)

    with open('relationship.json', 'w') as wf:
        for r in relationships:
            wf.write(json.dumps(relationships[r], ensure_ascii=False) + '\n')

def split_dataset():
    relationships = {}
    with open('relationship.json') as rf:
        for line in rf.readlines():
            r = json.loads(line)
            relationships[tuple(r['people'])] = r

    relationships = list(relationships.values())
    np.random.seed(1117)
    np.random.shuffle(relationships)
    train_relationships = relationships[:int(0.8 * len(relationships))]
    test_relationships = relationships[len(train_relationships):]

    with open('train_data.json', 'w') as wf:
        for r in train_relationships:
            # 删除重复句子
            r['sentences'] = list(r['sentences'])
            # 一对relationship取MAX_SEQLEN句话，超过则截断，分成多条数据
            if len(r['sentences']) > MAX_SEQLEN:
                np.random.sample(r['sentences'])
                n_new_entries = int(math.ceil(len(r['sentences']) / MAX_SEQLEN))
                for i in range(n_new_entries):
                    wf.write(json.dumps({
                        "sentences": r['sentences'][i * MAX_SEQLEN: min(len(r['sentences']), (i+1) * MAX_SEQLEN)],
                        "people": r['people'],
                        "label": r["label"]
                    }, ensure_ascii=False) + '\n')

            else:
                wf.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open('test_data.json', 'w') as wf:
        for r in test_relationships:
            # 删除重复句子
            r['sentences'] = list(r['sentences'])
            # 一对relationship取MAX_SEQLEN句话，超过则截断，分成多条数据
            if len(r['sentences']) > MAX_SEQLEN:
                np.random.shuffle(r['sentences'])
                n_new_entries = int(math.ceil(len(r['sentences']) / MAX_SEQLEN))
                for i in range(n_new_entries):
                    wf.write(json.dumps({
                        "sentences": r['sentences'][i * MAX_SEQLEN: min(len(r['sentences']), (i+1) * MAX_SEQLEN)],
                        "people": r['people'],
                        "label": r["label"]
                    }, ensure_ascii=False) + '\n')

            else:
                wf.write(json.dumps(r, ensure_ascii=False) + '\n')

def utf8_to_chinese(rpath, wpath):
    with open(wpath, 'w') as wf:
        with open(rpath) as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                content = json.loads(line)
                wf.write(json.dumps(content, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    #make_dataset()
    split_dataset()
    #utf8_to_chinese('train_data.json', 'train_data_chinese.json')