'''
This is the code to measure the informatio dependency between each feature and the label for tabular data.
Coded by Kevin Hu.
'''

import pandas as pd
import numpy as np
from sklearnex import patch_sklearn # for a faster computing if you are using Intel chip 
patch_sklearn() # you may comment out line 8 and line 9 if you haven't installed sklearnex
from sklearn.feature_selection import mutual_info_classif as mi
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial import distance


def fea_lable_muin(directory, standardize = False, plot_fig = True):
    '''
    ******************************************************
    Calulate the feature-label mutual information. 
    This fucntion resturns a vector, whose dimention is same as the feature space.
    Each value stands for the mutual information between that feature and the label, 
    i.e., how much information of label can be get with the observation of that feature?
    ******************************************************
    directory: the relative directory to the target data. 
            The data should be in the form of a matrix, where each row is a data sample.
            The leftmost column is the label, and start from column 1, each column stands for a feature / random varible.
            The data is preferred to be stored as the .csv file. You may modify the line 32 for reading different format.
    standardize: whether standardize the feature.
    plot_fig: whethre plot the result.
    ******************************************************
    '''
    df = pd.read_csv(directory, header=None)
    y = df.iloc[:,0]
    X = df.iloc[:,1:]
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    mutual_info = mi(X, y, discrete_features=False)
    if plot_fig:
        plt.plot(mutual_info)
        plt.xlabel('Feature')
        plt.ylabel('Bit')
        plt.show()
    return mutual_info

def muin_sim(directory_1, directory_2, standardize = False):
    '''
    ******************************************************
    Calculate the feature-label dependency similarity between two datasets.
    ******************************************************
    directory_1 and directory_2 are relative addresses of two datasets.
    ******************************************************
    '''
    mi_1 = fea_lable_muin(directory_1, standardize, False)
    mi_2 = fea_lable_muin(directory_2, standardize, False)
    sim = 1 - distance.cosine(mi_1, mi_2)
    print('The cosine similarity between two datasets is: ', sim)
    return sim



