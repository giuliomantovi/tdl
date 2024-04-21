import numpy as np
from operator import itemgetter


def four_highest_topics(lis, model):
    avg = 0
    if model == "pretrained":
        for item in lis:
            avg += item[1]
        avg /= len(lis)
    tuples = sorted(lis, key=lambda x: x[1], reverse=True)[:4]
    topics = []
    percentages = []
    for tupl in tuples:
        if model == "pretrained" and tupl[1] < avg:
            continue
        topics.append(tupl[0])
        percentages.append(round(tupl[1] * 1000, 2))

    # print(topics)
    # print(percentages)
    return topics, percentages
