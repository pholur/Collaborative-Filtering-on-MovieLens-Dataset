import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Project3_1234 import *
import csv
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import KNNWithMeans
from surprise.model_selection import *
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics

file_path = './ml-latest-small/ratings.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5), skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

def plotgraphs(x_axis, y_axis, nameofxaxis = 'x-axis', nameofyaxis = 'y-axis', title = 'Title', file_name = None):
    plt.plot(x_axis,y_axis)
    plt.xlabel(nameofxaxis, size = 15)
    plt.ylabel(nameofyaxis, size = 15)
    plt.title(title,size = 15)
    plt.show()
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')

#sim_options = {'name' : 'pearson' , 'user_based' : True} ############### user based #################
def problem_17_rmse_mae_full_dataset():
    x_axis = range(2,50,2)
    dim = len(x_axis)
    rmse_test_store = np.zeros(dim)
    mae_test_store = np.zeros(dim)

    for i in x_axis:
        algo = NMF(i, verbose=False) # i = number of latent factors
        result = cross_validate(algo, data, measures=['rmse', 'mae'], cv=10, verbose=True)

        rmse_score = np.mean(result['test_rmse'])
        mae_score = np.mean(result['test_mae'])

        ##################### Index to store values in rmse and mae ###################
        ind = int(i/2-1)

        rmse_test_store[ind] = rmse_score
        mae_test_store[ind] = mae_score


    pd.DataFrame(rmse_test_store).to_csv("rmse_test_store_10.csv")
    pd.DataFrame(mae_test_store).to_csv("mae_test_store_10.csv")

    plotgraphs(x_axis,rmse_test_store,'K','Mean rmse scores','Plot', 'q17_rmse.png')
    plotgraphs(x_axis,mae_test_store ,'K','Mean Mae scores' ,'Plot', 'q17_Mae.png' )


def problems_19_20_21_rmse_pop_unpop_hv():
    ratings = {}
    for r in data.raw_ratings:

        if r[1] not in ratings:
            ratings[r[1]] = []
        ratings[r[1]].append(r[2])

    ###############################################################################################

    popular_movies = [x for x in ratings if len(ratings[x]) > 2]
    unpopular_movies = [x for x in ratings if len(ratings[x]) <= 2]

    ###################################################################################
    kf = KFold(n_splits=10)
    rmse_popular_store = []
    for i in x_axis:

        algo = NMF(i, verbose=False)
        accu = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset) 
            test_trim = [x for x in testset if x[1] in popular_movies]
            predictions = algo.test(test_trim)
            accu.append(accuracy.rmse(predictions, verbose=True))
        s = np.mean(accu)
        rmse_popular_store.append(s)

    plotgraphs(x_axis,rmse_popular_store,'K','Mean RMSE scores','Plot of popular movies','q19_rmse_popular_movies.png')
    plotgraphs(x_axis,rmse_popular_store,'K','Mean RMSE scores','Plot of popular movies')

    ##########################################################################################

    kf = KFold(n_splits=10)
    rmse_unpopular_store = []
    for i in x_axis:

        algo = NMF(i, verbose=False)
        accu = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            test_trim = [x for x in testset if x[1] in unpopular_movies]
            predictions = algo.test(test_trim)
            accu.append(accuracy.rmse(predictions, verbose=True))
        s = np.mean(accu)
        rmse_unpopular_store.append(s)

    plotgraphs(x_axis,rmse_unpopular_store,'K','Mean RMSE scores','Plot of unpopular movies','q20_rmse_unpopular_movies.png')
    plotgraphs(x_axis,rmse_unpopular_store,'K','Mean RMSE scores','Plot of unpopular movies')

    ############ rates  "key" id, values are ratings #######################################
    movie_var = {}
    for k in ratings:
        # print(k)
        movie_var[k] = np.var(ratings[k])

    ####################################################################################
    highvar_movies = [ x for x in ratings if len(ratings[x])>= 5 and movie_var[x] >= 2 ]
    ##################################################################################

    kf = KFold(n_splits=10)
    rmse_highvar_store = []
    for i in x_axis:

        algo = NMF(i, verbose=False)
        accu = []
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            test_trim = [x for x in testset if x[1] in highvar_movies]
            predictions = algo.test(test_trim)
            accu.append(accuracy.rmse(predictions, verbose=True))
        s = np.mean(accu)
        rmse_highvar_store.append(s)

    pd.DataFrame(rmse_highvar_store).to_csv("rmse_highvar_store_21.csv")
    plotgraphs(x_axis,rmse_highvar_store  ,'K','Mean RMSE scores','Plot of high variance movies','q21_rmse_high_var_movies.png' )


##################  Problem 34 functions #######################

################## ROc plots #####################
def plot_roc(fpr, tpr,t, file_name=None):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', size=15)
    ax.set_ylabel('True Positive Rate', size=15)
    ax.legend(loc="lower right")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.title('Threshold  = %0.2f' % t, size=15)
    plt.show()
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')

def get_roc_params_KNN():
    sim_options = {'name' : 'pearson' , 'user_based' : True} ############### user based #################
    Threshold = [2.5, 3, 3.5, 4]
    best_k = 20
    trainset, testset = train_test_split(data, test_size=0.1, train_size=None, random_state=None, shuffle=True)
    algo = KNNWithMeans(k=best_k, sim_options=sim_options,verbose=True)
    algo.fit(trainset)
    predictions = algo.test(testset)
    trui = [ getattr(r,'r_ui') for r in predictions ]
    est = [ getattr(r,'est') for r in predictions ]

    fpr = {}
    tpr = {}
    t   = {}
    for i, t in enumerate(Threshold):
        trui_bin = [1 if r > t else 0 for r in trui]
        # est_bin = [1 if r > t else 0 for r in est]
        fpr[i], tpr[i], _ = metrics.roc_curve(trui_bin,est)
        plot_roc(fpr[i],tpr[i],t)
    return fpr, tpr, t

def get_roc_params_NMF():
    Threshold = [2.5, 3, 3.5, 4]
    best_k = 20
    trainset, testset = train_test_split(data, test_size=0.1, train_size=None, random_state=None, shuffle=True)
    algo = NMF(best_k, verbose=False)
    algo.fit(trainset)
    predictions = algo.test(testset)
    trui = [ getattr(r,'r_ui') for r in predictions ]
    est = [ getattr(r,'est') for r in predictions ]

    fpr = {}
    tpr = {}
    t   = {}
    for i, t in enumerate(Threshold):
        trui_bin = [1 if r > t else 0 for r in trui]
        # est_bin = [1 if r > t else 0 for r in est]
        fpr[i], tpr[i], _ = metrics.roc_curve(trui_bin,est)
        plot_roc(fpr[i],tpr[i],t)
    return fpr, tpr, t


# Need to add third algorithm 
fpr_knn, tpr_knn, t_knn = get_roc_params_KNN()
fpr_nmf, tpr_nmf, t_nmf = get_roc_params_NMF()

for i, thresh in enumerate(Threshold):
    fig, ax = plt.subplots()
    roc_auc_knn = auc(fpr_knn[i], tpr_knn[i])
    roc_auc_nmf = auc(fpr_nmf[i], tpr_nmf[i])
    ax.plot(fpr_knn[i], tpr_knn[i], lw=2, label='area under curve = %0.4f' % roc_auc_knn)
    ax.plot(fpr_nmf[i], tpr_nmf[i], lw=2, label='area under curve = %0.4f' % roc_auc_nmf)
    ax.grid(color='0.7', linestyle='--', linewidth=1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', size=15)
    ax.set_ylabel('True Positive Rate', size=15)
    ax.legend(loc="lower right")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.title('Threshold  = %0.2f' % thresh, size=15)
    plt.show()
