# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:15:40 2020

@author: dragu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:26:35 2020

@author: dragu
"""
# %%%
import os
 # 專案路徑
os.chdir("D:/python/doc2vec")

import jieba
import sys
import gensim
import sklearn
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence #從gensim導入doc2vec
TaggededDocument = gensim.models.doc2vec.TaggedDocument
#手動將'瓔珞'加入自定義userdict.txt中
jieba.load_userdict("./jieba/userdict.txt")

# 停止詞
stoplist = ['的','了','被','。','，','、','她','自己','他','並','和','都','去','\n']
# %%

#中文分詞
def  cut_files():
    filePath = 'data/rawdata.txt'
    fr = open(filePath, 'r', encoding="utf-8")
    fvideo = open('data/rawdata_jieba.txt', "w", encoding="utf-8")

    for line in fr.readlines():
        curLine =' '.join(list(jieba.cut(line)))
        fvideo.writelines(curLine)
    


def get_datasest():
    with open("data/rawdata_jieba.txt", 'r', encoding="utf-8") as cf:
        docs = cf.readlines()
        
        # 删除stopword
        for idx in list(range(0,len(docs))):
            docs[idx] = ' '.join([word for word in docs[idx].split( ) if word not in stoplist])
        docs = [doc for doc in docs if len(doc)>0]
        print(len(docs))

    x_train = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train

#訓練模型
def train(x_train, size=200, epoch_num=1):  # size=200 200維
	# 使用 Doc2Vec 建模
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    #model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model/model_dm_doc2vec')

    return model_dm


def test():
#    model_dm = Doc2Vec.load("model/model_dm_doc2vec")
    test_text = ['我', '喜歡', '傅恆']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    
    # 相似度前10
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims

# %%
if __name__ == '__main__':
    cut_files()
    x_train=get_datasest()
    model_dm = train(x_train)
    sims = test()
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print (words, sim, len(sentence[0]))
# %%  
        print(model_dm.similarity('瓔珞', '皇后'))
        print(model_dm.similarity('瓔珞', '皇上'))
#print(model_dm.wv.vocab)
# %%
'''
官方文件的基本範例
https://radimrehurek.com/gensim/models/doc2vec.html
'''
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
documents
# %%