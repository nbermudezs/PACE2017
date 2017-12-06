__author__ = 'Nestor Bermudez'
__mail__ = 'nab6@illinois.edu'

import pdb
import random

from scipy.sparse import csr_matrix

class PoiUserData:
    def __init__(self, user_id2idx, business_id2idx, idx2user_id, idx2business_id, user_ids, business_ids):
        self.user_id2idx = user_id2idx
        self.business_id2idx = business_id2idx
        self.idx2business_id = idx2business_id
        self.idx2user_id = idx2user_id
        self.matrix = csr_matrix(([1 for _ in range(len(user_ids)) ], (user_ids, business_ids)))
        self.user_ids = user_ids  # These are idxs
        self.business_ids = business_ids  # These are idxs

    def relations_count(self):
        return self.matrix.getnnz()

    def sample(self, percentage):
        count = self.relations_count()
        chosen = random.sample(range(count), int(count * percentage))
        return self._sample_from_indices(chosen)

    def _sample_from_indices(self, indices):
        nonzeros = self.matrix.nonzero()
        count = self.relations_count()

        new_business_id2idx = {}
        new_user_id2idx = {}
        new_idx2business_id = {}
        new_idx2user_id = {}
        new_user_idxs = []
        new_business_idxs = []

        for idx in indices:
            user_idx = self.user_ids[ nonzeros[ 0 ][ idx ] ]
            user_id = self.idx2user_id[ user_idx ]
            new_user_id2idx[ user_id ] = user_idx
            new_idx2user_id[ user_idx ] = user_id
            new_user_idxs.append(user_idx)

            business_idx = self.business_ids[ nonzeros[ 1 ][ idx ] ]
            business_id = self.idx2business_id[ business_idx ]
            new_business_id2idx[ business_id ] = business_idx
            new_idx2business_id[ business_idx ] = business_id
            new_business_idxs.append(business_idx)

        return PoiUserData(new_user_id2idx, new_business_id2idx, new_idx2user_id,
                           new_idx2business_id, new_user_idxs, new_business_idxs)

    def split(self, train_percent, val_percent, test_percent):
        total_percent = train_percent + val_percent + test_percent
        nonzeros = self.matrix.nonzero()
        count = self.relations_count()
        chosen = random.sample(range(count), int(count * total_percent))

        train_count = int(train_percent * count)
        val_count = int(val_percent * count)
        test_count = int(test_percent * count)

        training_data = self._sample_from_indices(chosen[ :train_count ])
        val_data = self._sample_from_indices(chosen[ (train_count + 1):train_count + val_count ])
        test_data = self._sample_from_indices(chosen[ (train_count + val_count + 1): ])

        print(' - split:')
        print('  - train: ' + str(train_count))
        print('  - eval: ' + str(val_count))
        print('  - test: ' + str(test_count))

        return (training_data, val_data, test_data)

    def get_user_ids(self):
        return list(self.user_id2idx.keys())

    def get_business_ids(self):
        return list(self.business_id2idx.keys())

if __name__ == '__main__':
    import pickle

    data_file = open('./poi-users-filtered.pkl', 'rb')
    data = pickle.load(data_file)
    pdb.set_trace()
    # data.sample(0.01)

    split_data = data.split(0.1, 0.1, 0.1)
    pdb.set_trace()
