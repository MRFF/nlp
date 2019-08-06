import re
from time import time
from opencc import OpenCC
import logging
import jieba
import pickle
from gensim import corpora
import gensim
from gensim.models import Word2Vec
import smart_open

# 装饰器，用以统计函数执行时间
def time_func(f):
    def wrapper(*args):
        start = time()
        print('Starting processing....')
        result = f(*args)
        end = time()
        print('Processing ended.....')
        duration = end -start
        print('----Processed in %ss----' % round(duration, 2))
        return result
    return wrapper

# 从中文维基数据源逐行提取文档，去除标签和符号，只保留中英文
@time_func
def strip_wiki_source(wiki_source):
    # 简繁体转换器
    convertor = OpenCC('t2s')

    output_file = open('wiki_stripped', 'w', encoding='utf-8')
    # 匹配<...>标签
    label_pattern = '<.+>'
    # 匹配各类中英文标点
    punc_pattern = '[“”，。（）\(\)·《》：:\-\"「」‘’？?!！,、；]'


    with open(wiki_source, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n': continue
            # 正则替换
            line = re.sub(label_pattern, '', line)
            line = re.sub(punc_pattern, '', line)
            # 由繁体转为简体
            simplified_line = convertor.convert(line)
            output_file.write(simplified_line)
    output_file.close()
    
    """
    Output Example:
    Starting processing....
    Processing ended.....
    ----Processed in 2205.73681807518s----
    """

# 读入清洗过的维基百科数据，逐行切词，序列化存入文件中，之后可再逐行读出
# 存入的数据每次读出都是一个列表，当中是切分过的句子，即[token_1, token_2,...token_n]
@time_func
def get_cut_lines(wiki_stripped):
    output_file = open('cut_lines', 'wb')
    with open(wiki_stripped, 'r', encoding='utf-8') as f:
        for line in f:
            # 空行跳过
            if line == '\n': continue
            # 切分后去掉行末的换行符
            cut_line = list(jieba.cut(line))[:-1]
            # 序列化，一次序列化一行，减少内存占用，之后读出时(pickle.load())也是一次读一行
            pickle.dump(cut_line,output_file, protocol=pickle.HIGHEST_PROTOCOL)
    output_file.close()

    """
    Output Example：
    Starting processing....
    Building prefix dict from the default dictionary ...
    Loading model from cache C:\\Users\\xyf22\\AppData\\Local\Temp\\jieba.cache
    Loading model cost 0.848 seconds.
    Prefix dict has been built succesfully.
    Processing ended.....
    ----Processed in 3395.708083152771s----
    """

# 读取切分好的数据，创建gensim dictionary和gensim corpus
# 对于Word2Vec训练，暂时用不上
@time_func
def create_gensim_dict_and_corpus(cut_lines):
    cut_file = open(cut_lines, 'rb')     
    dictionary = corpora.Dictionary()
    corpus = []
    processed = 0
    while True:
        try:
            processed += 1
            text = pickle.load(cut_file)
            dictionary.add_documents([text])
            corpus.append(dictionary.doc2bow(text))
            if not processed % 10000: print('----' + str(processed) + ' lines processed....')
            if processed > 2000000: break
        except EOFError:
            print('----End of file...')
            break
    print('----Starting Saving to gensim dict...')
    dictionary.save('wiki.dict')
    print('----Starting Saving to gensim corporus...')
    corpora.MmCorpus.serialize('wiki_corpus.mm', corpus)


# 训练Word2Vec模型，并保存
@time_func
def create_word2vec_model(cut_lines):
    cut_file = open(cut_lines, 'rb')
    sentences = []
    processed = 0
    threshhold = 2000000
    while True:
        try:
            processed += 1
            text = pickle.load(cut_file)
            sentences.append(text)
            if not processed % 10000: print('----' + str(processed) + ' lines processed....')
            if processed > threshhold: break
        except EOFError:
            print('----End of file...')
    print('----Training started...')
    model = Word2Vec(sentences=sentences, size=100, min_count=5, workers=4,)
    print('----Saving Word2Vec model to file...')
    model.save('wiki_word2vec.model')
    r"""
    Output Example:
    D:\Software\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
    warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    Starting processing....
    ----10000 lines processed....
    ----20000 lines processed....
    ----30000 lines processed....
    ----1990000 lines processed....
    ----2000000 lines processed....
    ----Training started...
    ----Saving Word2Vec model to file...
    Processing ended.....
    ----Processed in 886.01s----
    """

# 使用模型
def use_word2vec_model():
    model = Word2Vec.load('wiki_word2vec.model')
    # 和某个词最接近的5个词
    word_list = ['GDP', '爱迪生', '特斯拉', '孔乙己', '司马光',]
    print('5 most similar words with ' + word_list[0] + ' are:')
    for item in model.wv.most_similar(word_list[0], topn=5):
        print(item[0], item[1])

    # 判断两个词之间的相似度
    sim_w1 = word_list[1]
    sim_w2 = word_list[2]
    sim_w3 = word_list[3]
    sim1 = model.wv.similarity(sim_w1,sim_w2)
    sim2 = model.wv.similarity(sim_w1,sim_w3)
    print('\nThe similarity between {w1} and {w2} is {sim}'.format(w1=sim_w1, w2=sim_w2, sim=str(sim1)))
    print('The similarity between {w1} and {w2} is {sim}'.format(w1=sim_w1, w2=sim_w3, sim=str(sim2)))
    """
    Output Example:
    5 most similar words with GDP are:
    生产总值 0.8479525446891785
    经济总量 0.8407539129257202
    增速 0.8062247037887573
    工业产值 0.8035534620285034
    国民收入 0.7792089581489563

    The similarity between 爱迪生 and 特斯拉 is 0.6726107921940757
    The similarity between 爱迪生 and 孔乙己 is 0.03432663738711356
    """

# mm = corpora.MmCorpus('wiki_corpus.mm')

# strip_wiki_source('wikiraw')
# get_cut_lines('wiki_stripped')
# create_gensim_dict_and_corpus('cut_lines')
use_word2vec_model()

 