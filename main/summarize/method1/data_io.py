import numpy as np

import datetime

def get_word_map(wv):
    word_idx_dict = {word: idx for idx, word in enumerate(wv.index2entity)}
    return word_idx_dict
    
def getWordWeight(wv, a=1e-3):
    word2weight = {}
    '''
    return a word:weight dict
    '''

    print('1' + datetime.datetime.now().strftime('%H-%M-%S'))
    word_count_dict = {}
    for word in wv.vocab:
        word_count_dict[word] = wv.vocab[word].count
    N = float(sum(word_count_dict.values()))
    print('2' + datetime.datetime.now().strftime('%H-%M-%S'))
    if a <= 0: # when the parameter makes no sense, use unweighted
        a = 1.0
    for key, value in word_count_dict.items():
        word2weight[key] = a / (a + float(value)/N)

    return word2weight

def getWeight(word_idx_dict, word2weight):
    '''
    return a word_idx:word_weight dict
    '''
    weight4ind = {}
    for word, ind in word_idx_dict.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind
    
def lookupIDX(word_idx_dict,w):
    '''
    return the index of the word in vocab
    '''
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in word_idx_dict:
        return word_idx_dict[w]
    elif 'UUUNKKK' in word_idx_dict:
        return word_idx_dict['UUUNKKK']
    else:
        return len(word_idx_dict) - 1

def getSeq(sentence,word_idx_dict):
    '''
    Given a sentence, 
    the function returns a list containing indexs of each word in the vocab.
    '''
    sentence = sentence.split()
    X1 = []
    for word in sentence:
        X1.append(lookupIDX(word_idx_dict, word))
    return X1

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths) # pick up the max sentence length of given sentences
    x = np.zeros((n_samples, maxlen)).astype('int32') # initialize x matrix
    x_mask = np.zeros((n_samples, maxlen)).astype('float32') # initialize x_mask matrix
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask

def sentences2idx(sentences, word_idx_dict):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param word_idx_dict: a dictionary, word_idx_dict['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for sentence in sentences:
        seq1.append(getSeq(sentence, word_idx_dict))
    x1,m1 = prepare_data(seq1)
    return x1, m1

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight