import seaborn as sns
import numpy as np
import umap
from matplotlib import pyplot as plt
from utils import *
from sklearn.preprocessing import MinMaxScaler

def plot_phenotypes(data, score_dict):
    cluster_data, other_data = split_cluster_data(data)
    cluster_labels = list(cluster_data.keys())
    for cluster in set(cluster_labels):
        marker_dict = {}
        for marker in list(cluster_data.values())[0].columns:
            marker_dict[marker] = score_dict[marker][cluster]
        top_markers = sorted(marker_dict.keys(), key=lambda x: marker_dict[x])[-5:]
        scaler = MinMaxScaler().fit(other_data[cluster][top_markers])
        plt.figure(figsize=(9, 2))
        plt.subplot(121)
        plt.ylim((0,1))
        plt.xticks(rotation=30)
        plt.title('Top Markers in Cluster %s' % str(cluster))
        sns.violinplot(data=scale_frame(cluster_data[cluster][top_markers], scaler=scaler))
        plt.subplot(122)
        plt.ylim((0,1))
        plt.xticks(rotation=30)
        plt.title('Top Markers in Entire Data Set')
        sns.violinplot(data=scale_frame(other_data[cluster][top_markers], scaler=scaler))
        plt.show()

def plot_projection(x, y, labels):
    fig, ax = plt.subplots()
    for unique_label in set(labels):
        cluster_indices = [i for i, item in enumerate(labels) if item == unique_label]
        ax.scatter(x[cluster_indices], y[cluster_indices], label=unique_label)
    ax.legend()
    plt.show()

def main_plotting(data, score_dict):
    cluster_data, other_data = split_cluster_data(data)
    cluster_labels = list(cluster_data.keys())
    projection = umap.UMAP().fit_transform(data.drop("Labels", axis=1))
    while True:
        plot_projection(projection[:, 0], projection[:, 1], data['Labels'])
        cluster = int(input('Select cluster: '))
        if cluster not in cluster_labels:
            print('invalid')
            continue

        marker_dict = {}
        for marker in list(cluster_data.values())[0].columns:
            marker_dict[marker] = score_dict[marker][cluster]
        top_markers = sorted(marker_dict.keys(), key=lambda x: marker_dict[x])[-5:]
        scaler = MinMaxScaler().fit(other_data[cluster][top_markers])
        plt.figure(figsize=(9, 2))
        plt.subplot(121)
        plt.ylim((0, 1))
        plt.xticks(rotation=30)
        plt.title('Top Markers in Cluster %s' % str(cluster))
        sns.violinplot(data=scale_frame(cluster_data[cluster][top_markers], scaler=scaler))
        plt.subplot(122)
        plt.ylim((0, 1))
        plt.xticks(rotation=30)
        plt.title('Top Markers in Entire Data Set')
        sns.violinplot(data=scale_frame(other_data[cluster][top_markers], scaler=scaler))
        plt.show()
