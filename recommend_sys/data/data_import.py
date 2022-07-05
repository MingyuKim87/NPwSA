import os
import math
import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def movielens_10k_data(user_file_path, rating_file_path, item_file_path, trainigdata_ratio=0.8):
    '''
        Args:
            user_file_path : './data/movielens/ml-100k/u.user
            rating_file_path :  "./data/movielens/ml-100k/u.data
            item_file_path :  "./data/movielens/ml-100k/u.item

        Returns:
            trainig pd, val pd, test pd
    '''
    
    # user
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(user_file_path, sep='|', names=u_cols,
                    encoding='latin-1')

    # rating
    ratings = pd.read_csv(rating_file_path,
        sep="\t", names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # movie
    column_name = ["movie_id", "title", "release_date", "video_release_date",
              "imdb_url", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]

    movies = pd.read_csv(item_file_path,
        sep='|', names=column_name, usecols=range(24), encoding='latin-1')

    # create one merged DataFrame
    movie_ratings = pd.merge(ratings, movies, on="movie_id", how="left")
    lens = pd.merge(users, movie_ratings, on="user_id", how="right")
    column_list = ['user_id', 'rating', 'timestamp', 'age', 'sex', 'occupation', "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    column_list_2 = ['user_id', 'rating', 'timestamp', 'age', "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    column_list_3 = ['user_id', 'sex', 'occupation']
    
    # DF
    lens = lens[column_list]

    # DF Split
    user_movie_df = lens[column_list_2]
    # Treatement of Duplicated value 
    user_info_df = lens[column_list_3].drop_duplicates(['user_id'], keep='first')
    
    # Transform to one-hot vectors
    user_info_df = pd.get_dummies(user_info_df, columns=['sex', 'occupation'])
    
    # merge
    lens = pd.merge(user_movie_df, user_info_df, on='user_id', how='right')

    # filtering
    test_id = [5, 7, 10, 12, 15, 16, 29, 41, 43, 47, 49, 55, 61, 63, 64, 67,\
        73, 74, 77, 78, 79, 85, 87, 93, 120, 123, 130, 139, 148, 150, 158, 163,\
        176, 184, 185, 189, 191, 194, 195, 210, 212, 215, 217, 223, 226, 227, 228,\
        232, 234, 243, 245, 249, 253, 257, 258, 259, 264, 266, 267, 269, 270, 271,\
        277, 284, 291, 293, 298, 303, 305, 319, 321, 324, 329, 330, 339, 341, 344,\
        346, 350, 354, 363, 365, 369, 372, 381, 386, 387, 389, 391, 400, 403, 410,\
        412, 414, 423, 427, 434, 435, 439, 441, 443, 457, 461, 467, 469, 481, 490,\
        495, 498, 507, 508, 511, 516, 517, 524, 530, 544, 561, 563, 580, 591, 597,\
        607, 624, 629, 633, 635, 638, 641, 644, 662, 668, 678, 680, 692, 703, 711,\
        723, 730, 731, 740, 743, 751, 758, 764, 768, 770, 783, 785, 788, 789, 790,\
        793, 794, 798, 800, 801, 802, 804, 810, 812, 813, 814, 821, 827, 834, 836,\
        841, 843, 850, 851, 853, 860, 861, 868, 873, 887, 889, 893, 896, 901, 902,\
        903, 905, 906, 916, 921, 922, 923]

    # normalized
    normalized_timestamp = (lens['timestamp'] - lens['timestamp'].mean()) / lens['timestamp'].std()

    # delete and insert timestamp
    lens.drop(columns=['timestamp'], inplace=True)
    lens.insert(1, 'timestamp', normalized_timestamp)
    
    # permutation
        # indices
    indices = np.random.permutation(lens.shape[0])
    test_set_indices = np.where(np.isin(indices, test_id))
    train_val_set_indices = np.delete(indices, test_set_indices)

    train_val_data_count = train_val_set_indices.shape[0]
    train_data_count = math.floor(train_val_data_count * trainigdata_ratio)
    val_data_count = train_val_data_count - train_data_count

    train_id = train_val_set_indices[:train_data_count]
    val_id = train_val_set_indices[train_data_count:]

    # x and y
    target = lens['rating']
    context = lens.drop(columns=['rating'])

    # mask 1 (test)
    isin_filter = lens['user_id'].isin(test_id)
    test_pd_context = context[isin_filter]
    test_pd_target = target[isin_filter]

    # mask 2
    isin_filter = lens['user_id'].isin(train_id)
    train_pd_context = context[isin_filter]
    train_pd_target = target[isin_filter]

    # mask 2
    isin_filter = lens['user_id'].isin(val_id)
    val_pd_context = context[isin_filter]
    val_pd_target = target[isin_filter]

    return (train_pd_context, train_pd_target), \
        (val_pd_context, val_pd_target), \
        (test_pd_context, test_pd_target) 

if __name__ == "__main__":
    # DEBUG
    print(os.getcwd())
    
    
    user_file_path = './data/movielens/ml-100k/u.user'
    rating_file_path =  "./data/movielens/ml-100k/u.data"
    item_file_path =  "./data/movielens/ml-100k/u.item"

    train_pd, val_pd, test_pd = movielens_10k_data(user_file_path, rating_file_path, item_file_path)

    






    

