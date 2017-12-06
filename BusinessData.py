#!/usr/bin/python
'''Class used to load and save business data'''
__author__ = 'Haroun Habeeb, Nestor Bermudez'
__mail__ = 'hhabeeb2@illinois.com, nab6@illinois.edu'

import numpy as np
from scipy.sparse import csr_matrix

def haversine(lon1, lat1, lon2, lat2):
    """
    Taken from StackOverflow!!!!
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


class BusinessData:
    def __init__(self, business_id2idx, idx2business_id, business_edges):
        self.business_id2idx = business_id2idx  # string -> idx
        self.idx2business_id = idx2business_id  # int -> string
        self.business_edges = business_edges  # idx x idx -> True/False
        self.description = ''  # Provides some description of the data

    def get_idx(self, business_id):
        return self.business_id2idx[business_id]

    def get_business_id(self, idx):
        return self.idx2business_id[idx]

    def sample(self, business_ids):
        new_business_id2idx = {}
        new_idx2business_id = {}
        relations = []
        row_indices = []
        col_indices = []

        for (pos, i) in enumerate(business_ids):
            print('i', pos, end='\r')
            row_idx = self.business_id2idx[ i ]
            row = self.business_edges.getrow(row_idx)
            dense_row = row.todense()
            new_business_id2idx[ i ] = idx = self.business_id2idx[ i ]
            new_idx2business_id[ row_idx ] = i

            for j in business_ids:
                col_idx = self.business_id2idx[ j ]
                new_business_id2idx[ j ] = idx = self.business_id2idx[ j ]
                new_idx2business_id[ col_idx ] = j

                distance = dense_row[ 0, col_idx ]
                relations.append(distance)
                row_indices.append(row_idx)
                col_indices.append(col_idx)
        print(' - creating reduced sparse matrix')
        matrix = csr_matrix((relations, (row_indices, col_indices)))
        print(' - done creating reduced sparse matrix')
        return BusinessData(new_business_id2idx, new_idx2business_id, matrix)

if __name__ == '__main__':
    import pickle
    import pdb

    data_file = open('./BusinessData.0000 09.29.2017.data', 'rb')
    data = pickle.load(data_file)
    pdb.set_trace()


    import pdb
    idx2business_id = {
        0: 'abc',
        1: 'bcd',
        2: 'cde'
    }
    business_id2idx = {
        'abc': 0,
        'bcd': 1,
        'cde': 2
    }
    row_indices = [ 0, 0, 0, 1, 1, 1, 2, 2, 2 ]
    col_indices = [ 0, 1, 2, 0, 1, 2, 0, 1, 2 ]
    relations = [ 0, 1.22, 2.2, 1.22, 0, 0, 2.2, 0, 0 ]
    matrix = csr_matrix((relations, (row_indices, col_indices)))
    data = BusinessData(business_id2idx, idx2business_id, matrix)
    sampled = data.sample([ 'abc', 'bcd' ])
    pdb.set_trace()
