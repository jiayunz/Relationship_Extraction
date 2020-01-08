import wordcloud
from tqdm import tqdm
import json


font = '/System/Library/Fonts/PingFang.ttc'
stopwords = ['其他人物', '目标人物1', '目标人物2', '自己', '记者', '微博', '每周', '因为', '没有', '他们', '明星', '表示', '媒体', '报道', '已经', '对于', '一起', '导演', '一直', '香港', '台湾', '娱乐']
w = wordcloud.WordCloud(
    font_path=font,
    background_color='white',
    stopwords=stopwords,
    width=500,
    height=500

)
relationships = ['夫妻', '恋情', '前任', '好友', '家人', '搭档', '其他']
word_bags = {r: [] for r in relationships}

with open('relationship.json', 'r') as rf:
    lines = rf.readlines()
    for line in tqdm(lines, total=len(lines)):
        content = json.loads(line.strip())
        if content['label'] in ['绯闻', '情侣']:
            content['label'] = '恋情'
        if content['label'] in ['母子', '母女', '父女', '父子', '兄弟', '姐妹', '兄妹', '姐弟']:
            content['label'] = '家人'
        # if content['label'] in ['兄弟', '姐妹', '兄妹', '姐弟']:
        #    content['label'] = '兄弟姐妹'
        elif content['label'] not in relationships:
            continue

        word_bags[content['label']].extend(content['sentences'])

for rtype in relationships:
    w.generate(' '.join(word_bags[rtype]))
    w.to_file(rtype + '.png')