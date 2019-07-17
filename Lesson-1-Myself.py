from random import choice


def create_grammar(grammar_str, split='=>', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar


simple_grammar = """
sentence => noun_phrase verb_phrase
noun_phrase => Article Adj* noun
Adj* => null | Adj Adj*
verb_phrase => verb noun_phrase
Article =>  一个 | 这个
noun =>   女人 |  篮球 | 桌子 | 小猫
verb => 看着   |  坐在 |  听着 | 看见
Adj =>  蓝色的 | 好看的 | 小小的
"""

def generate(gram, target):
    if target not in gram:
        return target # terminal signal
    expanded = [generate(gram, t) for t in choice(gram[target])]
    return ''.join([e for e in expanded if e != 'null'])



#在西部世界里，一个”人类“的语言可以定义为：

human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 找找 | 想找点 
活动 = 乐子 | 玩的
"""


#一个“接待员”的语言可以定义为

host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""
# human_grammar = create_grammar(human, split='=')
# print(generate(human_grammar,'human'))
# ww_grammar = create_grammar(host, split='=')
# print(generate(ww_grammar, 'host'))

# import pandas as pd
# import pickle
# content = pd.read_csv(r'D:\Learning\NLP\data_source\news_original_source.csv', encoding='gb18030')
# articles = content['content'].tolist()
#
# import re
# def clean(string):
#     return re.findall('\w+', string)
#
#
# articles_cleaned = [''.join(clean(str(a))) for a in articles]
# # 要以二进制形式写入
# with open('articles.txt', 'wb') as f:
#     pickle.dump(articles_cleaned, f, protocol=pickle.HIGHEST_PROTOCOL)
#

import jieba
import pickle
#
# def cut(string): return list(jieba.cut(string))
#
#
# with open('articles.txt', 'rb') as f:
#     article_cleaned = pickle.load(f)
#
# article_words = [
#     cut(string) for string in article_cleaned
# ]
#
# with open('articles_words.txt', 'wb') as f:
#     pickle.dump(article_words, f, protocol=pickle.HIGHEST_PROTOCOL)

#
# from functools import reduce
# from operator import add
# with open('articles_words.txt', 'rb') as f:
#     articles_words = pickle.load(f)
# print(len(articles_words))
# 速度太慢
# articles_words_one_list = reduce(add, articles_words)
# articles_words_one_list = [word for item in articles_words for word in item ]
# with open('articles_words_one_list.txt', 'wb') as f:
#     pickle.dump(articles_words_one_list, f, protocol=pickle.HIGHEST_PROTOCOL)

from collections import Counter
with open('articles_words_one_list.txt', 'rb') as f:
    words = pickle.load(f)
words_count = Counter(words)
# 正态分布
# 求其对数
print(words_count.most_common(100))



def get_prob(w):
    return words_count[w] / len(words)
print(get_prob('牛肉'))
# 求词的联合概率
words_2_gram = [''.join(words[i:i+2]) for i in range(len(words[:-2]))]
words_2_count = Counter(words_2_gram)


def get_joint_prob(w1, w2):
    if w1 + w2 in words_2_count: return words_2_count[w1+w2] / len(words_2_gram)
    else:
        return   (get_prob(w1) + get_prob(w2)) / 2

# 求句子的联合概率
def get_prob_sentence(sentence):
    words = jieba.cut(sentence)
    sentence_prob = 1
    for i,_word in enumerate(sentence[:-1]):
        _next = sentence[i+1]
        _prob = get_joint_prob(_word, _next)
        sentence_prob *= _prob
    return sentence_prob


print(get_prob_sentence('早上吃了晚饭'))
print(get_prob_sentence('晚上吃了晚饭'))
sentnce_and_prob = []
for s in  [generate(create_grammar(simple_grammar), target='sentence') for i in range(10)]:
    print('sentence {} with prob: {}'.format(s, get_prob_sentence(s)))


def generate_best(n, grammar, target='sentence'):
    sentences = (generate(create_grammar(grammar), target) for i in range(n))
    probs = [get_prob_sentence(sentence) for sentence in sentences]
    best_prob = sorted(probs, reverse=True)[0]
    best_sentence = sentences[probs.index(best_prob)]
    print('sentence {} has the best prob: {}'.format(best_sentence, best_prob)
    return best_prob, best_sentence
