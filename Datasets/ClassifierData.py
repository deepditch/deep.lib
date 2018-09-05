import numpy as np
import itertools

def make_n_hot_labels(labels):
    classes = sorted(list(set(itertools.chain.from_iterable(labels))))
    label2idx = {v:k for k,v in enumerate(classes)}
    n_hot_labels = [np.zeros((len(classes),), dtype=np.float32) for l in labels]     
    for i, l in enumerate(labels):
        for classname in l:
            n_hot_labels[i][label2idx[classname]] = 1

    return n_hot_labels, classes