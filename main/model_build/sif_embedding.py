import os
import pandas as pd

from SIF.src import data_io
from SIF.src import params as sparams
from SIF.src import SIF_embedding
from utils.utils import remove_symbols, eval_vector_similar, time_counter


tokenized_file = '../../data/news_tokenized.csv'
word_embedding_file = '../data/no_symbol.word2vec'
weight_file = '../../data/total_vocab.txt'
similarity_file = '../data/similar_evaluation.txt'
test_article = '../data/test_article'

# tokenized_file = '../test_data/news_test_tokenized.csv'
# word_embedding_file = '../test_data/word_embedding_test.txt'
# saperated_file = '../test_data/total_test_contents_removed_symbols.txt'
# weight_file = '../test_data/appear_vocab.txt'
weight_para = 1e-3
rm_pc = 1


@time_counter
def map_initialize(word_embedding_file, weight_file, weight_para):
    (words2index, words_embedding) = data_io.getWordmap(word_embedding_file)
    word2weight = data_io.getWordWeight(weight_file, weight_para)
    weight4ind = data_io.getWeight(words2index, word2weight)
    return weight4ind, words2index, words_embedding


weight4ind, words2index, words_embedding = map_initialize(word_embedding_file, weight_file, weight_para)


def get_sentences_embedding(sentences):

    """
    return: embedding: ndarray, shape (n_samples, vector_space_dim)
    """

    sequence_matrix, mask_matrix = data_io.sentences2idx(sentences, words2index)
    weight_matrix = data_io.seq2weight(sequence_matrix, mask_matrix, weight4ind)
    params = sparams.params()
    params.rmpc = rm_pc

    embedding = SIF_embedding.SIF_embedding(words_embedding, sequence_matrix, weight_matrix, params)
    return embedding


def get_vectors_from_content(title, content, sentences):
    title_embedding = get_sentences_embedding([title])
    sentences_embedding = get_sentences_embedding(sentences)
    contents_embedding = get_sentences_embedding([remove_symbols(content.replace('\n', ' '))])

    return title_embedding, sentences_embedding, contents_embedding


@time_counter
def get_vectors_from_file():

    news_token_df = pd.read_csv(test_article)

    for i in news_token_df.index:
        sentences = [remove_symbols(sen) for sen in news_token_df.loc[i]['content'].split('\n')]
        title_embedding, sentences_embedding, contents_embedding = \
            get_vectors_from_content(news_token_df.loc[i]['title'],
                                     news_token_df.loc[i]['content'],
                                     sentences)
        index = 0
        sorted_similarity = []

        for sentence in sentences_embedding:
            content_sim = eval_vector_similar(sentence, contents_embedding[0])
            title_sim = eval_vector_similar(sentence, title_embedding[0])
            total_sim = (content_sim + title_sim) / 2
            sorted_similarity.append(([news_token_df.loc[i]['doc_id'], sentences[index], content_sim, title_sim, total_sim]))

            index += 1

        sorted_similarity = sorted(sorted_similarity, key=lambda x: float(x[4]), reverse=True)
        save(sorted_similarity)


def save(sorted_similarity):
    if not os.path.exists(similarity_file):
        with open(similarity_file, 'w') as sewh:
            sewh.write('doc_id,sentence,content_sim,title_sim,total_sim')
    with open(similarity_file, 'a') as sewh:
        for each in sorted_similarity:
            sewh.write('{},{},{},{},{}\n'.format(each[0], each[1].strip(), each[2], each[3], each[4]))


if __name__ == '__main__':
    get_vectors_from_file()



