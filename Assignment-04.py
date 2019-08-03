import re
from time import time
from opencc import OpenCC
import logging
import jieba
import pickle
 
# 装饰器，用以统计函数执行时间
def time_func(f):
    def wrapper(*args):
        start = time()
        print('Starting processing....')
        result = f(*args)
        end = time()
        print('Processing ended.....')
        duration = end -start
        print('----Processed in %ss----' % duration)
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
    输出样例：
    Starting processing....
    Processing ended.....
    ----Processed in 2205.73681807518s----
    """

# 读入清洗过的维基百科数据，逐行切词，序列化存入文件中，之后可再逐行读出
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
    输出样例：
    Starting processing....
    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\xyf22\AppData\Local\Temp\jieba.cache
    Loading model cost 0.848 seconds.
    Prefix dict has been built succesfully.
    Processing ended.....
    ----Processed in 3395.708083152771s----
    """

# strip_wiki_source('wikiraw') 
# get_cut_lines('wiki_stripped')        
