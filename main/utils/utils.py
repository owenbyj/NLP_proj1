import re
import datetime
import numpy as np


def remove_symbols(text):
    new_text = ''
    text_sentences = text.split('\n')
    for sentence in text_sentences:
        sentence_phrases = re.findall(r'[\w\s]+', sentence)
        new_text += ' '.join(sentence_phrases) + '\n'
    return new_text


def cal_vector_distance(vector1, vector2):
    inner_prod = (vector1 * vector2).sum()
    vctr1_norm = np.sqrt((vector1 * vector1).sum())
    vctr2_norm = np.sqrt((vector2 * vector2).sum())
    cos_distance = inner_prod / (vctr1_norm * vctr2_norm)
    return cos_distance


def eval_vector_similar(vector1, vector2):
    return (cal_vector_distance(vector1, vector2) + 1) / 2


def time_counter(func):

    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        start_formated_time = start_time.strftime("%Y-%M-%d %H:%M:%S")
        print("[START] - {} {}".format(func.__name__, start_formated_time))
        ret = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        end_formated_time = end_time.strftime("%Y-%M-%d %H:%M:%S")
        print("[COMPLETED] - {} {}".format(func.__name__, end_formated_time))
        timedelta = end_time - start_time
        print("[TIME CONSUMED] - {} {} hours".format(func.__name__, timedelta.seconds / 3600))
        return ret

    return wrapper
