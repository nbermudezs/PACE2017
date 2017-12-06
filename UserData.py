from __future__ import print_function
from scipy.sparse import csr_matrix

import pdb
import random

class UserData:
    def __init__(self, user_id2idx, idx2user_id, user_edges):
        self.user_id2idx = user_id2idx  # string -> idx
        self.idx2user_id = idx2user_id  # int -> string
        self.user_edges = user_edges  # idx x idx -> True/False
        self.description = ''  # Provides some description of the data

    def get_idx(self, user_id):
        return self.user_id2idx[user_id]

    def get_business_id(self, idx):
        return self.idx2user_id[idx]

    def sample(self, user_ids):
        new_user_id2idx = {}
        new_idx2user_id = {}
        relations = []
        row_indices = []
        col_indices = []

        for (pos, i) in enumerate(user_ids):
            print('i', pos, end='\r')
            row_idx = self.user_id2idx[ i ]
            row = self.user_edges.getrow(row_idx)
            dense_row = row.todense()
            new_user_id2idx[ i ] = idx = self.user_id2idx[ i ]
            new_idx2user_id[ row_idx ] = i

            for pos_j in range(pos - 1):
                j = user_ids[ pos_j ]
                col_idx = self.user_id2idx[ j ]
                new_user_id2idx[ j ] = idx = self.user_id2idx[ j ]
                new_idx2user_id[ col_idx ] = j

                value = dense_row[ 0, col_idx ]
                if value:
                    relations.append(value)
                    row_indices.append(row_idx)
                    col_indices.append(col_idx)
        print(' - creating reduced sparse matrix')
        matrix = csr_matrix((relations, (row_indices, col_indices)))
        print(' - done creating reduced sparse matrix')
        return UserData(new_user_id2idx, new_idx2user_id, matrix)

if __name__ == '__main__':
    import pickle
    with open('user_data.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        import pdb; pdb.set_trace()
        data.sample([ 'a', 'b' ])
