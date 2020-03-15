import re
import random
import pandas as pd

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from utils.utils import remove_symbols, time_counter

wiki_df = pd.read_csv('../test_data/wiki_test_tokenized.csv')
news_df = pd.read_csv('../test_data/news_test_tokenized.csv')

total_cont_path = '../../data/total_contents.txt'
total_cont_without_symbols_path = '../test_data/total_test_contents_removed_symbols.txt'

we_file_path = '../test_data/word_embedding_test.txt'

word_appearance_vocab_path = '../test_data/appear_vocab.txt'


def merge_all_contents():
    total_content_list = [str(content) for content in wiki_df['content']] + [str(content) for content in news_df['content']]
    total_contents = '\n'.join(total_content_list)

    total_contents_removed_symbols = remove_symbols(total_contents)

    with open(total_cont_path, 'w') as hf:
        hf.write(total_contents)

    with open(total_cont_without_symbols_path, 'w') as hf:
        hf.write(total_contents_removed_symbols)

    return total_contents, total_contents_removed_symbols


@time_counter
def random_mask(input_file):
    '''
    random mask words as <UNK>
    '''
    text = []
    output_file = "../data/merge_with_unk.txt"
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


@time_counter
def word2vec_training(text_file):
    sentences = LineSentence(text_file)
    model = Word2Vec(sentences, size=300, window=5, min_count=1, workers=16)
    model.wv.save("merge_with_unk.kv")
    # model.wv.save_word2vec_format("merge_with_unk_vector.txt", binary=False)
    return model


if __name__ == '__main__':
    modified_file = random_mask(total_cont_path)
    model = word2vec_training(modified_file)
