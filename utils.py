import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def between(num, bound1, bound2):
    return (num > bound1) and (num < bound2)

def get_indices(iterable, item):
    indices = []
    for i, entry in enumerate(iterable):
        if item == entry:
            indices.append(i)
    return indices

def split_cluster_data(all_data):
    labels = list(set(all_data['Labels']))
    cluster_data = {}
    other_data = {}
    for cluster in labels:
        cluster_data[cluster] = all_data[all_data['Labels'] == cluster].drop('Labels', 1)
        other_data[cluster] = all_data[all_data['Labels'] != cluster].drop('Labels', 1)
    return cluster_data, other_data

def scale_frame(frame, scaler=None, log=True):
    if log:
        frame = np.log1p(frame)
    if not scaler:
        scaler = MinMaxScaler().fit(frame)
    scaled_data = scaler.transform(frame)
    return pd.DataFrame(scaled_data, columns=frame.columns)

