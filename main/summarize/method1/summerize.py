from summarize.method1.sif_src import *
from summarize.method1.data_io import *
from summarize.method1.smooth import *
import pandas as pd
import re
import os
import jieba
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame
from gensim.models import KeyedVectors
from utils.utils import time_counter
from web.preloading import get_wv

data_path = os.getenv('DATA_PATH')
wv = get_wv()
# data_path = './data'
# data_path = '../../data'


class Params(object):

    def __init__(self, rmpc=1):
        self.LW = 1e-5
        self.LC = 1e-5
        self.eta = 0.05
        self.rmpc = rmpc

    def __str__(self):
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)


def ssplit(text):
    '''
    sentence split
    '''
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    return text


def read_doc(title, text):
    title = list(jieba.cut(title, use_paddle=True))
    title = " ".join(title)
    title = re.sub(" +", " ", title)
    content = []
    # with open(filename, "r", encoding="utf8") as fr:
    raw_text = ssplit(text.replace("\u3000", "")).split("\n")
    for line in raw_text:
        line = line.strip()
        if not line or line == "\n": continue
        line = list(jieba.cut(line, use_paddle=True))
        line = " ".join(line)
        line = re.sub(" +", " ", line)
        content.append(line)
    doc_dict = {title: content}
    return doc_dict


def mask_unk(seqs, wv):
    '''
    if OOV, mask the word with <UNK>
    '''
    result_seqs = []
    for sentence in seqs:
        word_list = sentence.split(" ")
        for idx, word in enumerate(word_list):
            if word not in wv.vocab:
                word_list[idx] = "<UNK>"
        result_seqs.append(" ".join(word_list))
    return result_seqs


def sif_emb(seqs, wv, params):
    '''
    get the emb of a sentence
    '''
    We = wv.vectors
    word_idx_dict = get_word_map(wv)
    x1, m1 = sentences2idx(seqs, word_idx_dict)
    weight4ind = getWeight(word_idx_dict, getWordWeight(wv, a=1e-3))
    weight_matrix = seq2weight(x1, m1, weight4ind)
    rmpc = 1
    params.rmpc = rmpc
    emb = SIF_embedding(We, x1, weight_matrix, params)
    return emb


def cos_sim_calculate(topic_vec, content_vec):
    '''
    calculate the cos sim between two vectors
    '''
    num = np.dot(topic_vec.T, content_vec)
    denom = np.linalg.norm(topic_vec) * np.linalg.norm(content_vec)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def raw_calculate(wv, params, doc_dict):

    for title, content in doc_dict.items():
        seqs = mask_unk([title], wv)
        seqs.extend(mask_unk(content, wv))
        emb = sif_emb(seqs, wv, params)
    topic = [" ".join(seqs[1:])]
    topic_emb = sif_emb(topic, wv, params)
    title_emb = emb[0]
    content_embs = emb[1:]
    sim_dict = {idx: cos_sim_calculate(topic_emb[0], value) for idx, value in enumerate(content_embs)}

    sim_sorted = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    return sim_sorted


def calculate_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def modify_score(score_list):
    '''
    modify the raw-calculate score
    '''
    result_list = []
    for idx, value in enumerate(score_list):
        if idx < 2:
            dis_range = score_list[:idx]
            dis_range.extend(score_list[idx + 1:idx + 3])
        elif len(score_list) - idx < 2:
            dis_range = score_list[idx - 2:idx]
            dis_range.extend(score_list[idx + 1:])
        else:
            dis_range = score_list[idx - 2:idx]
            dis_range.extend(score_list[idx + 1:idx + 3])
        avg_score_list = []
        for sur_v in dis_range:
            distance = calculate_distance(value, sur_v)
            score = 1 / (1 + distance)
            avg_score_list.append(score)
        result_list.append((value + np.mean(avg_score_list)) / 2)
    return result_list


def write_csv(text_content, sim_list):
    sim_data = list(zip(sim_list, text_content))
    csv_data = DataFrame(sim_data)
    csv_data.to_csv(os.path.join(data_path, "doc_sim.csv"), header=["score", "content"])


def get_score(input_title, input_text, smooth_type):
    # wv = KeyedVectors.load(os.path.join(data_path, 'merge_with_unk.kv'), mmap='r')
    # global wv
    params = Params()
    print('read_doc' + datetime.datetime.now().strftime('%H-%M-%S'))
    doc_dict = read_doc(input_title, input_text)
    print('raw_calculate' + datetime.datetime.now().strftime('%H-%M-%S'))
    raw_sim_sorted = raw_calculate(wv, params, doc_dict)
    print(datetime.datetime.now().strftime('%H-%M-%S'))
    tobe_smooth_list = []
    text_contents = list(doc_dict.values())[0]
    df_lists = [[ele[0], ele[1], text_contents[ele[0]].replace(" ", "")] for ele in raw_sim_sorted]

    smooth_types = {'smooth1': smooth1,
                    'smooth2': smooth2,
                    'compare': compare}
    if smooth_type not in smooth_type:
        raise TypeError("Inacceptable Smooth Type. valid values: 'smooth1', 'smooth2', 'compare'")
    return smooth_types[smooth_type](df_lists)


def compare(df_lists):
    return smooth1(df_lists) + smooth2(df_lists)


@time_counter
def smooth1(df_lists):
    x = [i[0] for i in df_lists]
    y_raw = [i[1] for i in df_lists]
    contents = [i[2] for i in df_lists]
    y_modified = modify_score(y_raw)
    final_sim_list = list(zip(x, y_modified, contents))
    final_sim_list = sorted(final_sim_list, key=lambda e: e[1], reverse=True)
    # plt.scatter(x, y_raw)
    # plt.scatter(x, y_modified)
    # plt.savefig(os.path.join(data_path, "compare.png"))
    top_rank = int(len(final_sim_list) / 4)
    smoothed_sim_list = final_sim_list[:top_rank]
    smoothed_sim_list = sorted(smoothed_sim_list, key=lambda e: e[0])
    return [{'smooth_method': 'smooth1', 'value': ''.join([i[2] for i in smoothed_sim_list])}]


@time_counter
def smooth2(df_lists):
    df_with_score = pd.DataFrame(df_lists, columns=['idx', 'score', 'content'])
    key_contents = get_smoothed_ranked_contents(df_with_score, 'kernel4')
    return [{'smooth_method': 'smooth2', 'value': ''.join(key_contents)}]


@time_counter
def summarize_from_text(input_title, input_text, smooth_type):
    return get_score(input_title, input_text, smooth_type)


@time_counter
def summarize_from_file(title, file_path, smooth_type):
    with open(file_path, 'r', encoding='utf-8') as rf:
        return summarize_from_text(title, rf.read(), smooth_type)


if __name__ == "__main__":
    print(summarize_from_file("林书豪怒怼美国政客种族歧视 为武汉抗疫捐款百万", os.path.join(data_path, 'test_article'), 'smooth1'))
    print('======================')
    print(summarize_from_file("林书豪怒怼美国政客种族歧视 为武汉抗疫捐款百万", os.path.join(data_path, 'test_article'), 'smooth2'))
