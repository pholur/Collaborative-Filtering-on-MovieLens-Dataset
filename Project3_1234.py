import pandas as pd
import numpy as np

def return_ratings_mat():
    # This function returns the matrix R that we use for everything in the project
    r = pd.read_csv("ml-latest-small/ratings.csv")
    r_prime = r.values

    max_user_id = max(r_prime[:,0])
    max_movie_id = max(r_prime[:,1])
    max_index = len(r.index)
    R = np.zeros((int(max_user_id),int(max_movie_id)), dtype=np.float)
    for i in range(0, max_index):
        R[int(r_prime[i,0]-1),int(r_prime[i,1]-1)] = r_prime[i,2]
    return R

if __name__ == '__main__':
     print return_ratings_mat()