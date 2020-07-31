#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" EER benchmarking """

__author__      = "Gaël Le Lan"
__copyright__   = "Copyright 2020, Gaël Le Lan"


from timeit import timeit

import numpy as np
from bob.measure import eer, eer_rocch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import make_scorer, roc_curve

from fast_eer import compute_eer, fast_eer
from sidekit.bosaris.detplot import rocch, rocch2eer


def benchmark_eer():
    rounds = 1
    pos_count = 100000
    neg_count = 9999999

    scale_p = np.random.uniform(low=0.1, high=2)
    scale_n = np.random.uniform(low=0.1, high=2)
    
    positive_scores = np.random.normal(loc=1, scale=scale_p, size=(pos_count,))
    negative_scores = np.random.normal(loc=-1, scale=scale_n, size=(neg_count,))

    neg_labels = -np.ones(negative_scores.shape, dtype=np.int)
    pos_labels = np.ones(positive_scores.shape, dtype=np.int)

    y_score = np.vstack((negative_scores.reshape(-1, 1), positive_scores.reshape(-1, 1))).flatten()
    y_true = np.vstack((neg_labels.reshape(-1, 1), pos_labels.reshape(-1, 1))).flatten()

    print('custom fast_eer \t\t{:.4f} sec \t {:.4f} %'.format(timeit(lambda:fast_eer(negative_scores, positive_scores), number=rounds), 100 * fast_eer(negative_scores, positive_scores)))

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    print('custom compute_eer \t\t{:.4f} sec \t {:.4f} %'.format(timeit(lambda:compute_eer(negative_scores, positive_scores), number=rounds), 100 * compute_eer(negative_scores, positive_scores)))

    print('bob eer \t\t\t{:.4f} sec \t {:.4f} %'.format(timeit(lambda:eer(negative_scores, positive_scores), number=rounds), 100 * eer(negative_scores, positive_scores)))

    pmiss, pfa = rocch(positive_scores, negative_scores)

    print('sidekit rocch \t\t\t{:.4f} sec'.format(timeit(lambda:rocch(positive_scores, negative_scores), number=rounds)))
    print('sidekit rocch2eer \t\t{:.4f} sec \t {:.4f} %'.format(timeit(lambda:rocch2eer(pmiss, pfa), number=rounds), 100 * rocch2eer(pmiss, pfa)))

    print('bob eer_rocch \t\t\t{:.4f} sec \t {:.4f} %'.format(timeit(lambda:eer_rocch(negative_scores, positive_scores), number=rounds), 100 * eer_rocch(negative_scores, positive_scores)))
    
    print('sklearn roc_curve \t\t{:.4f} sec'.format(timeit(lambda:roc_curve(y_true, y_score, pos_label=1), number=rounds)))
    print('sklearn eer \t\t\t{:.4f} sec \t {:.4f} %'.format(timeit(lambda:brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.), number=rounds), 100 * brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)))

if __name__ == "__main__":
    benchmark_eer()
