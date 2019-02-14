# local imports
from Project3_q30_33 import retrieve_data
from Project_3_q17_22_q34_functions import plotgraphs
from Project3_q30_33 import ret_user_dict
import time

# global imports
from surprise.model_selection import KFold
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise import KNNWithMeans

# global VARs
KNN_no_of_LF = 20
NMF_no_of_LF = 20
MF_no_of_LF = 20
sim_options = {'name' : 'pearson' , 'user_based' : True}


def ret_mod_user_dict(data):
    user_movies = {}
    for r in data.raw_ratings:
        if r[0] not in user_movies:
            user_movies[r[0]] = []
        if r[2] >= 3.0:
            user_movies[r[0]].append(r[1])
    return user_movies


def filter_test_set(testset, G_max, user_movie_dict, t):
    testfin = []
    user_movie_count = []

    for key in user_movie_dict:
        if len(user_movie_dict[key]) < t:
            user_movie_count.append(key)

    for test_iter in testset:
        if len(G_max[test_iter[0]]) > 0 and (test_iter[0] not in user_movie_count):
            testfin.append(test_iter)

    return testfin





def cross_val_(data, G_max, t, algo):

    pr = 0.0
    re = 0.0

    user_movie_dict = ret_user_dict(data)



    kf = KFold(n_splits=10)
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        testset = filter_test_set(testset, G_max, user_movie_dict, t)
        print testset
        predictions = algo.test(testset)
        time.sleep(10)
        print predictions
        time.sleep(100)


    return pr / 10.0, re / 10.0



if __name__ == '__main__':
    data = retrieve_data()
    G_max = ret_mod_user_dict(data)

    algo_NMF = NMF(NMF_no_of_LF, verbose=False)
    algo_SVD = SVD(n_factors=MF_no_of_LF)
    algo_KNN = KNNWithMeans(k=KNN_no_of_LF, sim_options=sim_options,verbose=False)

    # Q36
    Pr = []
    Re = []
    t = list(range(27))
    for l in t:
        Precision, Recall = cross_val_(data, G_max, l, algo_KNN)
        Pr.append(Precision)
        Re.append(Recall)

    plotgraphs(t, Pr, "Number of Suggestions", "Precision", "Precision Curve for KNN")
    plotgraphs(t, Re, "Number of Suggestions", "Recall", "Recall Curve for KNN")

    # Q37
    Pr = []
    Re = []
    for l in t:
        Precision, Recall = cross_val_(data, G_max, l, algo_NMF)
        Pr.append(Precision)
        Re.append(Recall)

    plotgraphs(t, Pr, "Number of Suggestions", "Precision", "Precision Curve for NNMF")
    plotgraphs(t, Re, "Number of Suggestions", "Recall", "Recall Curve for NNMF")

    # Q38
    Pr = []
    Re = []
    for l in t:
        Precision, Recall = cross_val_(data, G_max, l, algo_SVD)
        Pr.append(Precision)
        Re.append(Recall)

    plotgraphs(t, Pr, "Number of Suggestions", "Precision", "Precision Curve for MF")
    plotgraphs(t, Re, "Number of Suggestions", "Recall", "Recall Curve for MF")










