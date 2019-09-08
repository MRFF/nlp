import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from time import time


# 用以统计函数执行时间
def time_func(f):
    def wrapper(*args):
        start = time()
        print('Executing %s...' % f.__name__)
        print('--Start processing....')
        result = f(*args)
        end = time()
        print('--Process ended....')
        duration = end - start
        print('--Processed in %ss' % round(duration, 2))
        return result
    return wrapper

class Train(object):

    # 训练KNN模型
    # 需要定义静态方法，否则调用时第一个参数都是类本身
    @staticmethod
    @time_func
    def knn_train(x_train, y_train):
        """
        Evaluation:
        f1_score is: 0.7862440997977074
        accuracy is: 0.8111971411554497
        percision is: 0.9181102362204724
        recall is: 0.6875
        """
        from sklearn.neighbors import KNeighborsClassifier
        # 调参
        knn_clf = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform', algorithm = 'auto')
        # 训练模型
        return knn_clf.fit(x_train, y_train)

    @staticmethod
    @time_func
    def svm_train(x_train, y_train):
        """
        Evaluation:
        f1_score is: 0.9437652811735942
        accuracy is: 0.9452054794520548
        percision is: 0.9796954314720813
        recall is: 0.910377358490566
        """
        from sklearn import svm
        svm_clf = svm.SVC(gamma='scale')
        return svm_clf.fit(x_train, y_train)

    @staticmethod
    @time_func
    def lr_train(x_train, y_train):
        """
        Evaluation:
        f1_score is: 0.9384521633150518
        accuracy is: 0.9398451459201906
        percision is: 0.9709962168978562
        recall is: 0.9080188679245284
        """
        from sklearn.linear_model import LogisticRegression as LR
        return LR(random_state=0, solver='lbfgs',
                        multi_class='multinomial').fit(x_train, y_train)
    
    @staticmethod
    @time_func
    def naive_bayes_train(x_train, y_train):
        """
        Evaluation:
        f1_score is: 0.8186915887850468
        accuracy is: 0.8266825491363907
        percision is: 0.8678996036988111
        recall is: 0.7747641509433962
        """
        from sklearn.naive_bayes import GaussianNB
        bayes_clf = GaussianNB()
        return bayes_clf.fit(x_train.toarray(), y_train)

    @staticmethod
    @time_func
    def decision_tree_train(x_train, y_train):
        """
        Evaluation:
        f1_score is: 0.9591957421643997
        accuracy is: 0.958904109589041
        percision is: 0.9620403321470937
        recall is: 0.9563679245283019
        """
        from sklearn import tree
        dt_clf = tree.DecisionTreeClassifier()
        return dt_clf.fit(x_train, y_train)
        

# 评价模型
@time_func
def evaluate(clf, X, Y):
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_auc_score
    try:
        y_predicted = clf.predict(X)
    except TypeError:
        y_predicted = clf.predict(X.toarray())
    print('f1_score is: {}'.format(f1_score(Y, y_predicted)))
    print('accuracy is: {}'.format(accuracy_score(Y, y_predicted)))
    print('percision is: {}'.format(precision_score(Y, y_predicted)))
    print('recall is: {}'.format(recall_score(Y, y_predicted)))

def clean_cut(s):
    # 要将文本清洗、分词并以空格隔开，以便转为Tfidf向量表征
    # 清洗过程中发现有些文本还有\\n，也一并删除 
    return ' '.join(jieba.lcut(re.sub('[\r\n\u3000]', '', s).replace('\\n','')))


# 准备平衡的训练数据
@time_func
def prepare_inputs(df_train, feature):
    df_train[feature] = df_train[feature].fillna('').apply(clean_cut)
    df_train['is_xinhua'] = np.where(df_train['source'].str.contains('新华'), 1, 0)
    x_inputs = df_train[feature]
    y_inputs = df_train['is_xinhua']
    return x_inputs, y_inputs


# 避免训练数据不平衡，这里选取等量的两部分数据进行训练
def sample_train_data(df):
    df = df.fillna('')
    df_notxinhua = df[df.source != '新华社'][df.content != '']
    df_xinhua = df[df.source == '新华社']
    df_train = df_xinhua.sample(n=len(df_notxinhua))
    df_train = df_train.append(df_notxinhua)
    return df_train

# 找出潜在的抄袭作品，具体条件为预测来源为新华社，但实际又不是的
def find_potential_copies(y_test, y_predicted):
    potential_copy_num = [no for no, is_xinhua in enumerate(y_test) if is_xinhua == 0 and y_predicted[no] == 1]
    potential_copy_index = [y_test.index[no] for no in potential_copy_num]
    potential_copy = {df_train.source[p]: df_train.content[p] for p in potential_copy_index}
    return potential_copy





if __name__ == "__main__":
    df = pd.read_csv('news.csv', encoding='gb18030-2000')
    df_train = sample_train_data(df)
    x_inputs, y_inputs = prepare_inputs(df_train, 'content')
    # 必须要设置max_feature大一些，否则训练好后精确度很低
    vectorizer = TfidfVectorizer(max_features=600)
    # 要将文字转为tf-idf向量表示
    X = vectorizer.fit_transform(x_inputs.values)
    Y = y_inputs
    x_train, x_test, y_train, y_test = train_test_split(
        X , Y, train_size = 0.9, test_size=0.1
    )
    Train = Train()
    for item in dir(Train):
        if 'train' in item and not item.startswith('_'):
            train_func = getattr(Train, item)
            clf = train_func(x_train, y_train)
            evaluate(clf, x_test, y_test)