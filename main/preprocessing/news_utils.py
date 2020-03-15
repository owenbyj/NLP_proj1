import re
import os, sys
import pandas as pd
import random
import jieba


def news_tokenize(input_file, output_file):
    csv_data = pd.read_csv(input_file, usecols=["id", "title", "content"], encoding="gb18030")
    articles = csv_data["content"]
    titles = csv_data["title"].tolist()
    doc_ids = csv_data["id"].tolist()

    docs_dict = read_articles(articles)
    csv_list = []
    for idx, content in docs_dict.items():
        content_tokenized = []
        content = ssplit(content).split("\n")
        for line in content:
            if not line or line == "\n": continue
            line = " ".join(list(jieba.cut(line, use_paddle=True)))
            content_tokenized.append(line)
        doc_info_dict = {"doc_id": idx, "title": titles[idx], "content": "\n".join(content_tokenized)}
        csv_list.append(doc_info_dict)
    result_data = pd.DataFrame(csv_list, columns=["doc_id", "title", "content"])
    result_data.to_csv(output_file, index=False, sep=',')


def ssplit(text):
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    return text


def read_articles(articles):
    docs_idx_dict = {}
    for idx, content in enumerate(articles):
        if pd.isna(content):
            continue
        try:
            content = content.replace("\u3000", "").strip()
            content = re.sub("\t", " ", content)
            content = re.sub("\s+", " ", content)
            content = content.replace(r"\n", "\n").replace(r"\\n", "").replace("\r\n", "\n").replace("\n\n", "\n")
            if not content or content == "\n":
                continue
            docs_idx_dict[idx] = content
        except Exception as e:
            print(str(e) + "\n" + str(content))
    return docs_idx_dict


if __name__ == '__main__':
    news_tokenize("sqlResult_1558435.csv", "news_tokenized.csv")
