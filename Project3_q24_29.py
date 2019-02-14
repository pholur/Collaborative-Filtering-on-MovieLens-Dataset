import numpy as np
from Project3_q30_33 import trim, retrieve_data
from surprise.model_selection import KFold, cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.accuracy import rmse
import matplotlib.pyplot as plt


def Question24(data):
    ks = range(2,51,2)
    RMSE = []
    MAE = []
    for k in ks:
        model = SVD(n_factors=k)
        pred = cross_validate(model, data, cv=10)
        RMSE.append(np.mean(pred['test_rmse']))
        MAE.append(np.mean(pred['test_mae']))

    # Plot
    plt.plot(ks, RMSE)
    plt.xlabel('k')
    plt.ylabel('Average RMSE')
    plt.savefig('Q24_RMSE.png')
    plt.figure()
    plt.plot(ks, MAE)
    plt.xlabel('k')
    plt.ylabel('Average MAE')
    plt.savefig('Q24_MAE.png')

    
    index = np.argmin(RMSE)
    print("Best k: %i" % ks[index] )
    print("Lowest RMSE: %f" % RMSE[index] )
    print("Lowest MAE: %f" % np.min(MAE) )


def trimmed_test_MF(data, choice = 0):
    ks = range(2,51,2)
    avg_RMSEs = []
    for k in ks:
        kf = KFold(n_splits=10)
        rmse_total = 0
        for trainset, testset in kf.split(data):
            trimmed_testset = trim(data, testset, choice)
            model = SVD(n_factors=k).fit(trainset)
            pred = model.test(trimmed_testset)
            rmse_total += rmse(pred, verbose=False)
        rmse_total = rmse_total / 10.0
        avg_RMSEs.append(rmse_total)
        
    # Plot
    plt.plot(ks, avg_RMSEs)
    plt.xlabel('k')
    plt.ylabel('Average RMSE')
    plt.savefig('RMSE_' + str(choice) + '.png')

    index = np.argmin(avg_RMSEs)
    print("Best k: %i" % ks[index] )
    print("Lowest RMSE: %f" % avg_RMSEs[index] )



if __name__ == '__main__':
    data = retrieve_data()
    Question24(data)
    # Question 26
    print("Trimmed Test Set: Popular movies")
    trimmed_test_MF(data, 1)
    # Question 27
    print("\nTrimmed Test Set: Unpopular movies")
    trimmed_test_MF(data, 2)
    # Question 28
    print("\nTrimmed Test Set: High-variance movies")
    trimmed_test_MF(data, 3)


    





