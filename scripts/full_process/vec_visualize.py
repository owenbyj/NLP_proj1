# -*- coding:UTF-8 -*-
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.manifold import TSNE

def get_sample_words(wv,num):
    word_freq_dict = {}
    for word in wv.vocab:
        word_freq_dict[word] = wv.vocab[word].count
    word_freq_sorted = sorted(word_freq_dict.items(), key=lambda x:x[1], reverse=True)
    sample_words = [i[0] for i in word_freq_sorted][0:num]
    return sample_words
        
def tsne_plot(sample_words,wv,zhfont1):
    labels = []
    tokens = []
    for word in sample_words:
        tokens.append(wv.get_vector(word))
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=zhfont1
                     )
    plt.show()
    plt.savefig("visual.png")

if __name__ == "__main__":
    zhfont1 = matplotlib.font_manager.FontProperties(fname='SimHei.ttf')
    wv = KeyedVectors.load("merge_with_unk.kv",mmap="r")
    sample_words = get_sample_words(wv,500)
    tsne_plot(sample_words,wv,zhfont1)