import os

from gensim.models import KeyedVectors

data_path = os.getenv('DATA_PATH')


def get_wv():
    wv = KeyedVectors.load(os.path.join(data_path, 'merge_with_unk.kv'), mmap='r')
    return wv
