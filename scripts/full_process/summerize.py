from sif_src import *
from data_io import *
import pandas as pd
import re
import jieba
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import DataFrame

def ssplit(text):
    '''
    sentence split
    '''
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    return text

def read_doc(title, filename):
    title = list(jieba.cut(title,use_paddle=True))
    title = " ".join(title)
    title = re.sub(" +", " ", title)
    content = []
    with open(filename,"r", encoding="utf8") as fr:
        raw_text = ssplit(fr.read().replace("\u3000", "")).split("\n")
        for line in raw_text:
            line = line.strip()
            if not line or line == "\n": continue
            line = list(jieba.cut(line, use_paddle=True))
            line = " ".join(line)
            line = re.sub(" +", " ", line)
            content.append(line)
    doc_dict = {title: content}
    return doc_dict

def mask_unk(seqs,wv):
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
                
def sif_emb(seqs, wv,params):
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

def raw_calculate(wv,params,doc_dict):
    
    for title,content in doc_dict.items():
        seqs = mask_unk([title],wv)
        seqs.extend(mask_unk(content,wv))
        emb = sif_emb(seqs, wv, params)
    topic = [" ".join(seqs[1:])]
    topic_emb = sif_emb(topic, wv, params)
    title_emb = emb[0]
    content_embs = emb[1:]
    sim_dict = {idx:cos_sim_calculate(topic_emb[0],value) for idx,value in enumerate(content_embs)}
    
    sim_sorted = sorted(sim_dict.items(),key=lambda x:x[1], reverse=True)
    return sim_sorted
    
def calculate_distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def modify_score(score_list):
    '''
    modify the raw-calculate score
    '''
    result_list = []
    for idx,value in enumerate(score_list):
        if idx < 2:
            dis_range = score_list[:idx]
            dis_range.extend(score_list[idx+1:idx+3])
        elif len(score_list) - idx < 2:
            dis_range = score_list[idx-2:idx]
            dis_range.extend(score_list[idx+1:])
        else:
            dis_range = score_list[idx-2:idx]
            dis_range.extend(score_list[idx+1:idx+3])
        avg_score_list = []
        for sur_v in dis_range:
            distance = calculate_distance(value,sur_v)
            score = 1 / (1 + distance)
            avg_score_list.append(score)
        result_list.append((value + np.mean(avg_score_list)) / 2)
    return result_list

def write_csv(text_content, sim_list):
    sim_data = list(zip(sim_list, text_content))
    csv_data = DataFrame(sim_data)
    csv_data.to_csv("doc_sim.csv", header=["score", "content"])
    
if __name__ == "__main__":
    wv = KeyedVectors.load("merge.kv", mmap='r')
    params = params()
    doc_dict = read_doc("疫情","test.txt")
    raw_sim_sorted = raw_calculate(wv,params,doc_dict)
    knn_sim_dict = {}
    text_content = list(doc_dict.values())[0]
    x = [i[0] for i in raw_sim_sorted]
    y_raw = [i[1] for i in raw_sim_sorted]
    y_modified = modify_score(y_raw)
    final_sim_list = list(zip(x,y_modified))
    final_sim_list = sorted(final_sim_list, key=lambda x:x[1], reverse=True)
    write_csv(text_content, y_modified)
    plt.scatter(x,y_raw)
    plt.scatter(x,y_modified)
    plt.savefig("compare.png")
    result_idx = []
    for i in range(10):
        result_idx.append(final_sim_list[i][0])
        print(final_sim_list[i])
    for i, sent in enumerate(text_content):
        if i in result_idx:
            sent = sent.replace(" ", "")
            print(sent)



    