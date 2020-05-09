import tensorflow as tf

import cPickle
import numpy as np
import multiprocessing

from random import randint
from random import choice

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max
r=np.array([1,0,0,0,0,0,0,0,0,0])
print ndcg_at_k(r,6)
