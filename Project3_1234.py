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

    actual_no_of_movies = np.unique(r_prime[:,1]).size
    # actual_no_of_users = np.unique(r_prime[:,0]).size - same
    # print np.unique(r_prime[:,2])
    # print actual_no_of_users

    max_index = len(r.index)

    # Need to verify lowest rating != 0
    # print min(r_prime[:,2])
    R = np.zeros((int(max_user_id),int(max_movie_id)), dtype=np.float)

    for i in range(0, max_index):
        R[int(r_prime[i,0]-1),int(r_prime[i,1]-1)] = float(r_prime[i,2])
    return R, actual_no_of_movies

def sparsity(R, movie_num):
    return float(np.count_nonzero(R)) / (R.shape[0] * movie_num)

def plot_freq_ratings(R):
    # Plots
    plt.hist(np.ravel(R), bins = [0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25])
    # Emphasizes the disparity of the ratings
    plt.xlabel("Rating Bins")
    plt.ylabel("Frequency of Bin")
    plt.show()

def plot_movie_rate_freq_desc(R, movie_num):
    R_sub = (R > 0.0).astype(int)
    Count_R_sub = np.sum(R_sub, axis = 0)

    Index = np.argsort(Count_R_sub)
    Index = Index[::-1]
    Index = Index[0:movie_num]

    x = [i for i in range(0,movie_num)]
    plt.plot(x,Count_R_sub[Index],label=str(Index))
    plt.xlabel("Index of Movie (10 users marked")
    plt.ylabel("Number of Ratings")
    order = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000]
    plt.xticks(order,Index[order]+1)
    plt.show()

def plot_user_vote_freq_desc(R):
    R_sub = (R > 0.0).astype(int)
    Count_R_sub = np.sum(R_sub, axis = 1)

    Index = np.argsort(Count_R_sub)
    Index = Index[::-1]

    x = [i for i in range(0,len(Count_R_sub))]
    plt.plot(x,Count_R_sub[Index],label=str(Index))
    plt.xlabel("Index of User (7 users marked)")
    plt.ylabel("Number of Ratings")
    order = [0,100,200,300,400,500,600]
    plt.xticks(order,Index[order]+1)
    plt.show()

if __name__ == '__main__':
    R, movie_num = return_ratings_mat()
    # print R.shape
    # print sparsity(R, movie_num) # Q1
    # plot_freq_ratings(R) # Q2
    # plot_movie_rate_freq_desc(R, movie_num) # Q3
    # plot_user_vote_freq_desc(R) # Q4
