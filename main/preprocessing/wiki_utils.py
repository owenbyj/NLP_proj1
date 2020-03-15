# -*- coding: utf-8
import re
import os, sys
import json
from pyhanlp import *
# from stanfordcorenlp import StanfordCoreNLP
import jieba
import time
from lxml import etree
import requests


def ssplit(text):
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    return text


def wiki_tokenize(input_file, output_file):
    start = time.time()
    doc_id_counter = 0
    csv_list = []
    with open(input_file, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip().replace("\u3000", "")
            line = re.sub("\s+", " ", line)
            if not line or line == "\n": continue
            if line.startswith("<doc id="):
                doc_id_counter += 1
                doc_dict = {}
                doc_content = []
                doc_info = HanLP.convertToSimplifiedChinese(line)
                doc_info = etree.HTML(doc_info)
                doc_dict["doc_id"] = doc_info.xpath("//@id")[0]
                doc_dict["title"] = doc_info.xpath("//@title")[0]
                continue
            elif line == "</doc>":
                doc_dict["content"] = "\n".join(doc_content)
                csv_list.append(doc_dict)
            else:
                line = HanLP.convertToSimplifiedChinese(line)
                line = ssplit(line)
                for sentence in line.split("\n"):
                    if not sentence or sentence == "\n": continue
                    sentence = sentence.strip()
                    sentence_tok = list(jieba.cut(sentence, use_paddle=True))
                    doc_content.append(" ".join(sentence_tok))
    end = time.time()
    print(f"{end - start}")

    import pandas as pd
    result_data = pd.DataFrame(csv_list, columns=["doc_id", "title", "content"])
    result_data.to_csv(output_file, index=True, sep=',')


if __name__ == '__main__':
    wiki_tokenize("wiki_coding_conv.txt", "wiki_tokenized_2.csv")
