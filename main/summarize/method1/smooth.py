import numpy as np
import pandas as pd


def smooth_kernel_4(sentenceArray):
    out = []
    cnnArray = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
    for i in range(1, len(sentenceArray) - 2):
        c = np.dot(sentenceArray[i - 1:i + 3], cnnArray)
        # print(sentenceArray[i-1:i+3],c)
        if len(out) == 0:
            out = out + [round(c, 3)]
        if len(out) == len(out) == len(sentenceArray) - 3:
            out = out + [round(c, 3)] + [round(c, 3)]
        out = out + [round(c, 3)]
        # print(out)
    # out.append((sentenceArray[-1]+sentenceArray[-2])/2)
    return out


def get_smoothed_ranked_contents(data_frame, kernel):
    processed = data_frame
    # processed["new_score_3"] = smooth_kernel_3(processed["score"])
    processed["new_score_4"] = smooth_kernel_4(processed["score"])
    # processed["new_score_5"] = smooth_kernel_5(processed["score"])

    processed.columns = ['idx', 'score', 'content', 'kernel4']
    processed.sort_values(kernel, inplace=True, ascending=False)
    # sort by kernel score

    top_ranked = int(len(processed) / 4)  ##select the sentenses with top 1/4 importance
    tmp = processed[0:top_ranked]
    tmp.sort_values("idx", inplace=True)
    return tmp["content"]


