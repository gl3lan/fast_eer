# -*- coding: utf-8 -*-

""" Fast EER computation """

__author__ = "Gaël Le Lan"
__copyright__ = "Copyright 2020, Gaël Le Lan"
__license__ = "Apache 2.0"

import numpy as np


def compute_eer(negative_scores, positive_scores):
    """Linear complexity EER computation

    Args:
        negative_scores (numpy array): impostor scores
        positive_scores (numpy array): genuine scores

    Returns:
        float: Equal Error Rate (EER)
    """

    neg_labels = -np.ones(negative_scores.shape, dtype=np.int)
    pos_labels = np.ones(positive_scores.shape, dtype=np.int)

    scores = np.vstack((negative_scores.reshape(-1, 1), positive_scores.reshape(-1, 1))).flatten()
    t_nt = np.vstack((neg_labels.reshape(-1, 1), pos_labels.reshape(-1, 1))).flatten()

    sort_indices = np.argsort(-scores)
    sorted_scores = scores[sort_indices]
    sorted_t_nt = t_nt[sort_indices]

    t_count = np.sum(t_nt > 0)
    nt_count = np.sum(t_nt < 0)

    m_count = t_count
    fa_count = 0
    eer = 100
    far_frr_predicate = 100
    eer_threshold = 0

    last_score = -np.Inf
    m_count_increment = 0
    fa_count_increment = 0

    jdx = 0
    for idx, score in enumerate(sorted_scores):
        if score != last_score:
            m_count += m_count_increment
            fa_count += fa_count_increment
            m_count_increment = 0
            fa_count_increment = 0
        # target
        if sorted_t_nt[idx] == 1:
            if abs((m_count) / t_count - fa_count / nt_count) <= far_frr_predicate:
                far_frr_predicate = abs((m_count) / t_count - fa_count / nt_count)
                eer_threshold = score
                eer = ((m_count) / t_count + fa_count / nt_count)/2
            else:
                break
            m_count_increment -= 1
            jdx += 1
        # non-target
        elif sorted_t_nt[idx] == -1:
            if abs((m_count) / t_count - fa_count / nt_count) <= far_frr_predicate:
                far_frr_predicate = abs((m_count) / t_count - fa_count / nt_count)
                eer_threshold = score
                eer = ((m_count) / t_count + fa_count / nt_count)/2
            else:
                break
            fa_count_increment += 1
            jdx += 1
        last_score = score
    return eer

def fast_eer(negatives, positives):
    """Logarithmic complexity EER computation

    Args:
        negative_scores (numpy array): impostor scores
        positive_scores (numpy array): genuine scores

    Returns:
        float: Equal Error Rate (EER)
    """

    positives = np.sort(positives)
    negatives = np.sort(negatives)[::-1]

    pos_count = positives.shape[0]
    neg_count = negatives.shape[0]

    p_score = positives[0]
    n_score = negatives[0]

    p_index = 0
    n_index = 0

    next_p_jump = pos_count//2
    next_n_jump = neg_count//2

    kdx = 0
    while True:
        kdx += 1
        if p_index < 0 or n_index < 0:
            return 0
        if p_index > pos_count or n_index > neg_count:
            return 100
        if p_score < n_score:
            p_index = p_index + next_p_jump
            n_index = n_index + next_n_jump
            if next_p_jump == 0 and next_n_jump == 0:
                break
        elif p_score >= n_score:
            p_index = p_index - next_p_jump
            n_index = n_index - next_n_jump
            if next_p_jump == 0 and next_n_jump == 0:
                break
                
        p_score = positives[p_index]
        n_score = negatives[n_index]
        next_p_jump = next_p_jump//2
        next_n_jump = next_n_jump//2

    eer_predicate = 100

    tfr = (abs(p_index))/pos_count
    tfa = (1+abs(n_index))/neg_count
    if (p_score == n_score and tfr == tfa):
        return tfr

    while positives[p_index] < negatives[n_index]:
        if p_index < pos_count - 1:
            p_index += 1
        elif n_index < neg_count - 1:
            n_index += 1
        else:
            break

    while positives[p_index] > negatives[n_index] and n_index >= 1:
        n_index -= 1

    tfr = (1+p_index)/pos_count
    tfa = (1+n_index)/neg_count

    while tfa > tfr:
        p_index += 1
        while positives[p_index] > negatives[n_index] and n_index >= 1:
            n_index -= 1
        tfr = (1+p_index)/pos_count
        tfa = (1+n_index)/neg_count

    if abs(tfr - tfa) <= eer_predicate:
        eer_predicate = abs(tfr - tfa)
        eer = (tfr + tfa) / 2
    else:
        return eer

    tfr = p_index/pos_count
    tfa = (1+n_index)/neg_count
    if abs(tfr - tfa) <= eer_predicate:
        eer_predicate = abs(tfr - tfa)
        eer = (tfr + tfa) / 2
    else:
        return eer

    while True:
        while negatives[n_index + 1] <= positives[p_index - 1]:
            p_index -= 1
            tfr = p_index/pos_count
            tfa = (1+n_index)/neg_count
            if abs(tfr - tfa) <= eer_predicate:
                eer_predicate = abs(tfr - tfa)
                eer = (tfr + tfa) / 2
            else:
                return eer
        while negatives[n_index + 1] > positives[p_index - 1]:
            n_index += 1
            tfr = p_index/pos_count
            tfa = (1+n_index)/neg_count
            if abs(tfr - tfa) <= eer_predicate:
                eer_predicate = abs(tfr - tfa)
                eer = (tfr + tfa) / 2
            else:
                return eer

    return eer
