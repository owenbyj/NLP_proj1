# -*- coding: utf-8 -*-
import os, sys
import re
import random
import pandas as pd
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def random_mask(input_file):
    '''
    random mask words as <UNK>
    '''
    text = []
    output_file = "merge_with_unk.txt"
    with open(input_file, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(" +", " ", line)
            text.append(line)
    line_sample = random.sample(range(len(text)), len(text) // 10)
    for line_idx in line_sample:
        line = text[line_idx].split(" ")
        word_idx = random.choice(range(len(line)))
        line[word_idx] = "<UNK>"
        text[line_idx] = " ".join(line)
    with open(output_file, "w", encoding="utf8") as fw:
        fw.write("\n".join(text))
    return output_file


def word2vec_training(text_file):
    sentences = LineSentence(text_file)
    model = Word2Vec(sentences, size=300, window=5, min_count=1, workers=16)
    model.wv.save("merge_with_unk.kv")
    # model.wv.save_word2vec_format("merge_with_unk_vector.txt", binary=False)
    return model

if __name__ == "__main__":
    text_file = "merge_content.txt"
    modified_file = random_mask(text_file)
    model = word2vec_training(modified_file)

