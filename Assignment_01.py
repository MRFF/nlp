import re
import jieba
import pandas
from random import choice
from collections import Counter

data_folder = r'D:\Learning\NLP\data_source'
host_grammar_str = """
sentence = 寒暄 报数 询问 业务相关 结尾 
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


# 生成语法
def create_grammar(grammar_str, split='=>', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar


def generate(gram, target):
    if target not in gram:
        return target  # terminal signal
    expanded = [generate(gram, t) for t in choice(gram[target])]
    return ''.join([e for e in expanded if e != 'null'])


def generate_sentences(n, grammar, target='sentence'):
    sentences = (generate(create_grammar(grammar), target) for i in range(n))
    return sentences


# 处理文本，生成语料
def get_insurance_text():
    insurance_file = data_folder + r'\insurance_questions_answers.txt'
    with open(insurance_file, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()
    sentences = [re.findall('\w+', t.split('++$++')[2].split()[0])[0] for t in text_lines if t]

    return concat_strlist(sentences)


def get_movie_comment_text():
    movie_comments_file = data_folder + r'\movie_comments.csv'
    content = pandas.read_csv(movie_comments_file, encoding='utf-8')
    comments = content['comment'].tolist()
    texts = [re.findall('\w+', c) for c in comments if type(c) == str]
    text_list = [t for item in texts for t in item ]
    return concat_strlist(text_list)


def concat_strlist(str_lst):
    _text = ''
    for s in str_lst:
        _text += s
    return _text


# 语言模型
def get_words_count():
    insurance_text = get_insurance_text()
    comment_text = get_movie_comment_text()
    combined_text = insurance_text + comment_text
    # cut返回的是generator
    cut_list = list(jieba.cut(combined_text))
    words_count = Counter(cut_list)
    return list(cut_list), words_count


def get_prob(counter, w, words):
     return counter[w] / len(words)


def get_joint_prob(w1, w2, words_two_gram, counter, words):
    if w1 + w2 in words_two_gram: return words_two_gram[w1+w2] / len(words_two_gram)
    else:
        return (get_prob(counter, w1, words) + get_prob(counter, w2, words)) / 2


def get_prob_sentence(sentence):
    words = list(jieba.cut(sentence))
    sentence_prob = 1
    for i,_word in enumerate(sentence[:-1]):
        _next = sentence[i+1]
        _prob = get_joint_prob(_word, _next, words_two_gram, words_count, words)
        sentence_prob *= _prob
    return sentence_prob


# 生成最佳句子
def generate_best(n, grammar_str, target='sentence'):
    sentences = [generate(create_grammar(grammar_str, split='='), target) for i in range(n)]
    probs = [get_prob_sentence(sentence) for sentence in sentences]
    best_prob = sorted(probs, reverse=True)[0]
    best_sentence = sentences[probs.index(best_prob)]
    print('sentence {} has the best prob: {}'.format(best_sentence, best_prob))
    return best_prob, best_sentence


words, words_count = get_words_count()
words_two_gram = Counter([''.join(words[i:i+2]) for i in range(len(words[:-2]))])
generate_best(10, grammar_str=host_grammar_str, target='sentence')
