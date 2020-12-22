import os
import sys
import argparse
import itertools
import numpy as np
import random
import math
import pandas as pd
import time
# from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.tsa.stattools import pacf
# from pyentrp import entropy as ent
# from PermEnt import weighted_ordinal_patterns as WPEnt


def min_weighted_perm_entropy(ts, min_order=2, max_order=5, min_delay=1, max_delay=3, normalize=True):
    min_ent = 1000
    o  = min_order
    d = min_delay
    for i in range(min_order, max_order+1):
        for j in range(min_delay, max_delay+1):
            entropy = weighted_perm_entropy(ts, order=i, delay=j, normalize=True, normalizeType='observed')
            if entropy < min_ent:
                min_ent = entropy
                o = i
                d = j
    return o, d


def filter_dropout(data,dropout_rate):
    '''
    :param data: in terms of total count per day
    :param dropout_rate: probability of deleting a data point
    :return: processed data
    '''
    filtered_data = np.zeros(len(data))
    for ind in range(len(data)):
        dec, inte = math.modf(data[ind])
        for count in range( int( inte ) ):
            dice = np.random.uniform()
            if dice < (1-dropout_rate):
                filtered_data[ind] += 1
        if dec!=0:
            dice = np.random.uniform()
            if dice < (1 - dropout_rate):
                filtered_data[ind] += dec

    return filtered_data

def get_rmse(true_ts, pred_ts):
    mse = mean_squared_error(true_ts, pred_ts, multioutput='raw_values')[0]
    rmse = np.sqrt(mse)
    return rmse

def weighted_ordinal_patterns_fast(ts, embdim, embdelay=1):
    time_series = ts
    # possible_permutations = list(itertools.permutations(range(embdim)))
    N = len(time_series) - embdelay * (embdim-1)
    weights = np.zeros(N)
    for i in range(N):
        Xi = time_series.iloc[range(i,i+embdim*embdelay,embdelay)]
        Xi_mean = np.mean(Xi)
        Xi_var = (Xi-Xi_mean)**2
        weight = np.mean(Xi_var)
        weights[i] = weight
    weights = weights/np.sum(weights) # normalization

    return weights

def _embed(x, order=3, delay=1):
    """Time-delay embedding.

    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.

    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def weighted_perm_entropy(time_series,order=3,delay=1, normalize=False, normalizeType='normal'):
                           
    """Permutation Entropy.

    Parameters
    ----------
    time_series : list or np.array
        Time series
    weights : np.array
        weight of each subsequence
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        weighted Permutation Entropy

    """
    x = np.array(time_series)
    weights = weighted_ordinal_patterns_fast(time_series,order,delay)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort') # axis = -1
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1) # axis = 1 (second axis)
    # Return the counts
    _, c = np.unique(hashval, return_inverse=True)
    Types = np.unique(c)
    p = np.zeros(len(Types))
    for elem in Types:
        ind = np.where(c == elem)[0]
        p[elem] = np.sum( weights[ind] )

    # Use np.true_divide for Python 2 compatibility
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        if normalizeType=='observed':
            pe /= np.log2( len(Types) )
            # print('Normalized by Amount of Observed Permutation')
        else:
            pe /= np.log2(math.factorial(order))
    return pe

def weighted_ordinal_patterns(ts, embdim, embdelay=1):
    time_series = ts
    # possible_permutations = list(itertools.permutations(range(embdim)))
    temp_list = list()
    wop = list()
    for i in range(len(time_series) - embdelay * (embdim - 1)):
        # Xi = time_series[i:(embdim+i)]
        Xi = time_series.iloc[range(i,i+embdim*embdelay,embdelay)]
        # Xn = time_series[(i+embdim-1): (i+embdim+embdim-1)]
        Xi_mean = np.mean(Xi)
        Xi_var = (Xi-Xi_mean)**2
        weight = np.mean(Xi_var)
        sorted_index_array = list(np.argsort(Xi))
        temp_list.append([''.join(map(str, sorted_index_array)), weight])
    result = pd.DataFrame(temp_list,columns=['pattern','weights'])
    total_weight = np.sum(result['weights'].get_values())
    result['weights'] = result['weights']/total_weight # normalization
    # pattern_list = dict(result['pattern'].value_counts())
    for pat in (result['pattern'].unique()):
        wop.append( np.sum(result.loc[result['pattern']==pat,'weights'].values) )
    return wop # list of weights corresponding to different patterns

def permutated_entropy(ts, embdim=3, embdelay=1,normalize=False):
    probDist = weighted_ordinal_patterns(ts, embdim, embdelay)
    if isinstance(probDist,list):
        probDist = np.array(probDist)
    entropy = -np.dot(probDist,np.log2(probDist))
    if normalize:
        entropy /= np.log2(math.factorial(embdim))
    return entropy


def main():
    # random.seed(1234)
    length = int(365)
    # cushion = 7
    output_result = True

    os.chdir("/Users/leonardohuang/Desktop/research_archive/ISI/data3")
    data_filepath = "data_(5,0,1)_3.txt"
    exogs_filepath = "exogs_(3,0,2).txt"
    print('Current Path: ', os.getcwd())
    print('Loading Ground Truth Data from: ', data_filepath)
    print('Loading External Signal Data from: ', exogs_filepath)
    data = np.loadtxt(data_filepath)
    external_signal = np.loadtxt(exogs_filepath)

    L = len(data)
    data1 = np.random.randint(0,20,L)
    data2 = np.random.randint(0,50,L)
    data3 = np.random.randint(0,100,L)
    data4 = np.zeros(L)
    data4[0] = 50
    c     = 2
    for t in range(L-1):
        data4[t+1] = data4[t] + np.random.uniform(-c,c,1)

    synthetic_data_all = [data]
    # synthetic_data_all = [data2,data4]

    wiggle =1e-2
    dropout_grid = np.append( np.linspace(wiggle,0.15,30) , np.linspace(0.15+wiggle,1-wiggle,40) )

    permEnt_Dim3 = np.zeros(len(dropout_grid))
    permEnt_Dim4 = np.zeros(len(dropout_grid))
    permEnt_Dim5 = np.zeros(len(dropout_grid))
    permEnt_Dim6 = np.zeros(len(dropout_grid))
    permEnt_Dim7 = np.zeros(len(dropout_grid))

    # repeat = 5
    repeat = 60
    Delay = 1
    normalizeType = 'Actual'
    # flag = 'normal'
    flag = 'weighted'
    for data_ind in range(len(synthetic_data_all)):
        if data_ind == 0:
            use_external_signal = False
        if data_ind == 1:
            use_external_signal = True

        print('\n\n\n Dataset No.', data_ind + 1)

        synthetic_data = synthetic_data_all[data_ind]
        train = synthetic_data[int(5 / 12 * length):int(11 / 12 * length)]
        # exog_train = external_signal[int(5 / 12 * length):int(11 / 12 * length)]

        # Define external signals
        # if use_external_signal:
        #     print('This experiment uses external signal. \n')
        #     exogData_train = exog_train
        # else:
        #     print('This experiment doe NOT use external signal. \n')
        #     exogData_train = None

        for ind in np.array(range(len(dropout_grid))):
            dropout_rate = dropout_grid[ind]
            print("\n Currently evaluating dropout rate:", dropout_rate)

            print(
                'Data Set {} out of {} for dropout rate {}'.format(data_ind + 1, len(synthetic_data_all), dropout_rate))

            train_filtered = filter_dropout(train, dropout_rate=dropout_rate)
            rep_ind = 0
            temp3 = temp4 = temp5 = temp6 = temp7 = 0
            if flag == 'weighted':
                while rep_ind < repeat:
                    temp3 += weighted_perm_entropy(train_filtered, 3, Delay, normalize=True)
                    temp4 += weighted_perm_entropy(train_filtered, 4, Delay, normalize=True)
                    temp5 += weighted_perm_entropy(train_filtered, 5, Delay, normalize=True)
                    temp6 += weighted_perm_entropy(train_filtered, 6, Delay, normalize=True)
                    temp7 += weighted_perm_entropy(train_filtered, 7, Delay, normalize=True)

                    rep_ind += 1
            elif flag == 'normal':
                while rep_ind < repeat:
                    temp3 += ent.permutation_entropy(train_filtered, 3, Delay, normalize=True)
                    temp4 += ent.permutation_entropy(train_filtered, 4, Delay, normalize=True)
                    temp5 += ent.permutation_entropy(train_filtered, 5, Delay, normalize=True)
                    temp6 += ent.permutation_entropy(train_filtered, 6, Delay, normalize=True)
                    temp7 += ent.permutation_entropy(train_filtered, 7, Delay, normalize=True)

                    rep_ind += 1
            permEnt_Dim3[ind] = temp3 / repeat
            permEnt_Dim4[ind] = temp4 / repeat
            permEnt_Dim5[ind] = temp5 / repeat
            permEnt_Dim6[ind] = temp6 / repeat
            permEnt_Dim7[ind] = temp7 / repeat

        fig1, ax1 = plt.subplots()
        ax1.plot(dropout_grid, permEnt_Dim3, color='y',label='dim = 3')
        ax1.plot(dropout_grid, permEnt_Dim4, color='r',label='dim = 4')
        ax1.plot(dropout_grid, permEnt_Dim5, color='g',label='dim = 5')
        ax1.plot(dropout_grid, permEnt_Dim6, color='b',label='dim = 6')
        ax1.plot(dropout_grid, permEnt_Dim7, color='m',label='dim = 7')
        ax1.set_xlabel('Dropout Rate')
        ax1.set_ylabel('Weighted Permutation Entropy')
        ax1.legend(loc='best')
        plt.show()
        fig1.savefig("_PermEnt.pdf")

        if output_result:
            os.chdir('/Users/leonardohuang/Desktop/research_archive/ISI/data3')
            filename = '_PermEnt_dropout_' + str(dropout_rate) + time.strftime('%Y%m%d_%H%M%S') + normalizeType + '.txt'
            f = open(filename, 'w')
            title = 'DropoutRate' + '\t' + 'PermEntDim3' + '\t' + 'PermEntDim4' + '\t' + 'PermEntDim5' + '\t' + \
                        'PermEntDim6' + '\t' 'PermEntDim7' + '\t' + '\n'
            f.write(title)
            for printind in range(len(dropout_grid)):
                entity = str(dropout_grid[printind]) + '\t' + str(permEnt_Dim3[printind]) + '\t' + str(permEnt_Dim4[printind]) +\
                             '\t' + str(permEnt_Dim5[printind]) + '\t' + str(permEnt_Dim6[printind]) + '\t' +\
                             str(permEnt_Dim7[printind]) + '\n'
                f.write(entity)
            f.close()

if __name__ == "__main__":
    # main(sys.argv)
    # load_external_signals("d2web")
    main()
