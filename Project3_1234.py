import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def return_ratings_mat():
    # Returns the matrix R that we use for everything in the project
    # Rij is the rating by user i in movie j
    r = pd.read_csv("ml-latest-small/ratings.csv")
    r_prime = r.values

    max_user_id = max(r_prime[:,0])
    max_movie_id = max(r_prime[:,1])
    max_index = len(r.index)

    # Need to verify lowest rating != 0
    # print min(r_prime[:,2])
    R = np.zeros((int(max_user_id),int(max_movie_id)), dtype=np.float)

    for i in range(0, max_index):
        R[int(r_prime[i,0]-1),int(r_prime[i,1]-1)] = float(r_prime[i,2])
    return R

def sparsity(R):
    return float(np.count_nonzero(R)) / (R.shape[0] * R.shape[1])

def plot_freq_ratings(R):
    # Plots
    print np.ravel(R)
    plt.hist(np.ravel(R), bins = [0,0.5,1,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])

    # Emphasizes the disparity of the ratings
    plt.yscale('log', nonposy='clip')
    plt.xlabel("Rating Bins")
    plt.ylabel("Frequency of Bin (log scale)")
    plt.show()


if __name__ == '__main__':
    R = return_ratings_mat()
    print R
    print sparsity(R) # Q1
    plot_freq_ratings(R) # Q2