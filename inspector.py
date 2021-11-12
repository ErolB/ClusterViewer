import pickle as pkl
from metrics import *
from utils import *
from visualize import *

metric = 'vi'

if __name__ == '__main__':
    # load and preprocess
    with open('AllData_training_withLabels_adt_0.8.pkl', 'rb') as pkl_file:
        data = pkl.load(pkl_file)
    cluster_labels = data['Labels']

    # calculation and plotting
    kl_scores = rank_kl(data, cluster_labels)
    #plot_phenotypes(data, kl_scores)
    main_plotting(data, kl_scores)

