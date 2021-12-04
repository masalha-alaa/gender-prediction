import numpy as np
from sklearn.feature_selection import SelectKBest


def select_k_best_2(vocabulary, features_df, labels, k_best):
    """
    Select K best features using sklearn's SelectKBest, which is based on the ANOVA F-value test:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    :return: Best K features (unordered)
    """
    best_k_words = SelectKBest(k=k_best)
    best_k_words.fit(features_df, labels)
    best_k_words_idx = best_k_words.get_support(indices=True)
    best_k_words_list = vocabulary[best_k_words_idx]

    return {'Classifier-Irrelevant': best_k_words_list}


def select_k_best(coef, feature_names, k_best):
    """
    Select K best features using the classifier's importance ratings.
    :return: Best K features in a descending order.
    """
    # assuming coefficients were scaled.
    # Source: https://stackoverflow.com/a/34052747/900394
    # partition the importances array, such that the pivot is at the last -k elements
    # (in simpler words: we get the largest argmax k elements at the end)
    importances = coef[0]
    k_top_argmax = np.argpartition(importances, -k_best)[-k_best:]
    # sort them in descending order according to their importances
    top_k_sorted = k_top_argmax[np.argsort(importances[k_top_argmax])][::-1]
    best_k_features = feature_names[top_k_sorted]

    return best_k_features
