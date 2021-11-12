from scipy.stats import gaussian_kde
from scipy.stats import entropy
from scipy.integrate import quad
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils import *

def kl_div(dist1, dist2):
    kde1 = gaussian_kde(dist1)
    kde2 = gaussian_kde(dist2)
    def kl_util(i):
        return kde1(i) * np.log(kde1(i) / kde2(i))
    upper_bound = max(max(dist1), max(dist2))
    return quad(kl_util, a=0, b=upper_bound)

# functions for calculating variation of information

def r_f(labels1, labels2, i, j):
    if len(labels1) != len(labels2):
        print('failure')
        return
    indices1 = get_indices(labels1, i)
    indices2 = get_indices(labels2, j)
    intersection = [item for item in indices1 if item in indices2]
    return len(intersection) / len(labels1)

def v_i(labels1, labels2):
    summation = 0
    for i in set(labels1):
        for j in set(labels2):
            r_value = r_f(labels1, labels2, i, j)
            p_i = list(labels1).count(i) / len(labels1)
            q_j = list(labels2).count(j) / len(labels1)
            entry = -r_value * (np.log(r_value/p_i) + np.log(r_value/q_j))
            if not np.isnan(entry) and not np.isinf(entry):
                summation += entry
    return summation

# ranking functions

def rank_vi(data):
    start_labels = AgglomerativeClustering(n_clusters=10).fit_predict(data)
    v_dict = {}  # global VI
    cluster_dict = {}  # cluster integrity
    for marker in data.columns:
        print(marker)
        new_data = data.drop(marker, axis=1)
        new_clustering = AgglomerativeClustering(n_clusters=10).fit_predict(new_data)
        v_dict[marker] = v_i(start_labels, new_clustering)
        # calculate cluster integrity
        temp = {}
        for cluster in set(start_labels):
            original_indices = get_indices(start_labels, cluster)
            temp[cluster] = entropy(new_clustering[original_indices])
        cluster_dict[marker] = temp
    return v_dict, cluster_dict

def rank_kl(data, cluster_labels):
    cluster_dict = {}
    data = np.arcsinh(data.drop('Labels', axis=1))
    for marker in data.columns:
        print(marker)
        temp = {}
        for cluster in set(cluster_labels):
            cluster_data = data.iloc[[i for i, item in enumerate(cluster_labels) if item==cluster]]
            kl = kl_div(cluster_data[marker], data[marker])
            temp[cluster] = kl[0]
        cluster_dict[marker] = temp
    return cluster_dict