'''
    File containing Dataset classes for pytorch PACE model
'''
__author__ = 'Haroun Habeeb, Nestor Bermudez'
__mail__ = 'hhabeeb2@illinois.edu, nab6@illinois.edu'


import os
import pickle
import time
import pdb
import torch
import numpy as np
from scipy.sparse import find
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Three classes for data handling really.
from UserData import UserData
from PoiUserData import PoiUserData
from BusinessData import BusinessData
from attribute_data import AttributeData


class YelpDataset(Dataset):
    '''
        A Dataset class used for datasets for our
        implementation of PACE
        Parameters
        ----------

        Methods
        -------
    '''

    def __init__(self,
                 data_directory,
                 relations_file='poi-users-filtered-09112017.pkl',
                 user_graph_file='user_data.pkl',
                 business_graph_file='BusinessData.0000 09.29.2017.data',
                 negative_samples=5,
                 is_cuda=False,
                 is_hetero=False,
                 attribute_file='attributes0000.11.22.2017.pkl'):
        '''
            ARGS:
                data_directory:
                    str
                    Directory containing the datafiles
                relations_file:
                    str
                    filename of pickle file containing user-business relations
                user_graph_file:
                    str
                    filename of pickle file containing information about users
                business_graph_file:
                    str
                    filename of pickle file containing information about users
                is_hetero:
                    bool
                    represents whether to use Heterogeneous data or not.
                attribute_file:
                    str
                    file that contains an object of class
        '''
        self.negative_samples = negative_samples
        self.relation_visit_count = {}
        tic = time.time()
        with open(os.path.join(data_directory, user_graph_file), 'rb') as f:
            self.user_data = pickle.load(f, encoding='latin1')
        print('Loading user graph took', time.time() - tic, ' seconds')
        tic = time.time()
        with open(os.path.join(data_directory,
                               business_graph_file), 'rb') as f:
            self.business_data = pickle.load(f)
            self.business_data.idx2business_id = {v: k for k, v in
                                                  self.business_data.
                                                  business_id2idx.items()}
        print('Loading business graph took', time.time() - tic, ' seconds')
        tic = time.time()
        with open(os.path.join(data_directory, relations_file), 'rb') as f:
            self.poi_user_data = pickle.load(f, encoding='latin1')
        print('Loading poi_user_data took', time.time() - tic, ' seconds')

        # #### print some statistics
        self.n_relations = self.poi_user_data.relations_count()
        self.n_users = self.poi_user_data.matrix.shape[0]
        self.n_businesses = self.poi_user_data.matrix.shape[1]
        self.n_full_user_graph = self.user_data.user_edges.getnnz()
        self.n_full_business_graph = self.business_data.business_edges.getnnz()
        print('\n')
        print('Number of relations=', self.n_relations)
        print('Number of users=', self.n_users)
        print('Number of businesses=', self.n_businesses)
        self.is_cuda = is_cuda
        self.is_hetero = is_hetero

        self.user_idxs = [ self.user_data.user_id2idx[uid]
                           for uid in self.poi_user_data.idx2user_id.values() ]
        self.business_idxs = [ self.business_data.business_id2idx[uid]
                           for uid in self.poi_user_data.idx2business_id.values() ]

        # Handling Heterogeneous data.
        if is_hetero:
            with open(os.path.join(data_directory, attribute_file), 'rb') as f:
                self.attribute_data = pickle.load(f)
        # Don't need to unify data
        # Keep self.business_data fixed
        # try:
        # bd_idx2pu_b_idx = [self.poi_user_data.business_id2idx[self.business_data.idx2business_id[i]] for i in range(0, self.business_data.business_edges.shape[0])]
        # pu_u_idx2ud_idx = [self.user_data.user_id2idx[self.poi_user_data.idx2user_id[i]] for i in range(0, self.poi_user_data.matrix.shape[0])]
        # self.poi_user_data.matrix = self.poi_user_data.matrix[:, bd_idx2pu_b_idx]
        # self.user_data.user_edges = self.user_data.user_edges[pu_u_idx2ud_idx, pu_u_idx2ud_idx]

        # self.poi_user_data.business_id2idx = self.business_data.business_id2idx
        # self.poi_user_data.idx2business_id = self.business_data.idx2business_id
        # self.user_data.user_id2idx = self.poi_user_data.user_id2idx
        # self.user_data.idx2user_id = self.poi_user_data.idx2user_id
        # except:
        #     pdb.set_trace()

    def __len__(self):
        return self.negative_samples * self.poi_user_data.relations_count()

    def __getitem__(self, virtual_relation_idx, raw=True):
        '''
            generator that fetches one item from the dataset
            Args
            ------
            relation_idx : idx of relation to sample

            Returns
            -------
            user_id (str): one user
            business_id (str): one business
            user_idx (int): idx in self.poi_user_data for user_id
            business_idx (int): idx in self.poi_business_data for business_id
            prediction (int): 1 if user_id-business_id relation exists.
                              0 otherwise.
            user_context (sparse? 1d array): friendships of the users
            business_context (sparse? 1d array):
                businesses close to the business you're looking at
        '''
        # relation_idx = relation_idx % (self.n_relations * self.negative_samples)
        # if relation_idx in negative_sample_count:
        #     negative_sample_count = self.relation_visit_count[relation_idx]
        #     negative_sample_idx = self.negative_sample_idx_es[relation_idx]
        # else:
        #     negative_sample_idx = self.negative_sample_idx_es[relation_idx] = np.random.randint(0, self.negative_samples)
        #     negative_sample_count = self.relation_visit_count[relation_idx] = 0

        # if not(negative_sample_count == negative_sample_idx):  # Generate a negative sample for relation_idx
        relation_idx = (virtual_relation_idx // self.negative_samples) % self.n_relations
        negative_sample = virtual_relation_idx % self.negative_samples
        if not(negative_sample == 0):
            if (np.random.randint(2) == 0):  # Negative sample the user_id
                business_idx = self.poi_user_data.business_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
                # business_idx = self.poi_user_data.business_id2idx[business_id]
                # Avoid "incorrect data" ... TODO: Think about do I need to do this?
                user_idx = np.random.randint(0, self.n_users)
                while self.poi_user_data.matrix[user_idx, business_idx] != 0:
                    user_idx = np.random.randint(0, self.n_users)
                # user_id = self.poi_user_data.idx2user_id[user_idx]
            else:  # Negative sample the business_id
                user_idx = self.poi_user_data.user_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
                # user_idx = self.poi_user_data.user_id2idx[user_id]

                # Avoid "incorrect data" ... TODO: Think about do I need to do this?
                business_idx = np.random.randint(0, self.n_businesses)
                while self.poi_user_data.matrix[user_idx, business_idx] != 0:
                    business_idx = np.random.randint(0, self.n_businesses)
                # business_id = self.poi_user_data.idx2business_id[business_idx]
            prediction = 0 if raw else torch.LongTensor([[0]])
        else:  # Get a sample from the data using relation_idx
            user_idx = self.poi_user_data.user_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
            # user_idx = self.poi_user_data.user_id2idx[user_id]
            business_idx = self.poi_user_data.business_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
            # business_idx = self.poi_user_data.business_id2idx[business_id]
            # relation_idx += 1
            prediction = 1 if raw else torch.LongTensor([[1]])

        # self.relation_visit_count[relation_idx] += 1
        # self.relation_visit_count[relation_idx] = self.relation_visit_count[relation_idx] % (self.negative_samples)

        # user_data_idx = self.user_data.user_id2idx[user_id]
        # business_data_idx = self.business_data.business_id2idx[business_id]

        # TODO: needs more processing, this will have all 1m users instead of the 70k poi-users
        # it should be of size n_users (to match the user_context_length in PACE.py)
        user_context = self.user_data.user_edges[user_idx].A[:, self.user_idxs]
        if np.count_nonzero(user_context) == 0:
            user_context = np.zeros(self.n_users) if raw else torch.zeros(1, self.n_users).long()
        else:
            user_context = np.greater(user_context, 0).astype(int) if raw else torch.from_numpy(user_context).gt(0).long()

        business_context = self.business_data. \
            business_edges[business_idx].A[:, self.business_idxs]
        if np.count_nonzero(business_context) == 0:
            business_context = np.zeros(self.n_businesses) if raw else torch.zeros(1, self.n_businesses).long()
        else:
            business_context = np.greater(business_context, 0).astype(int) if raw else torch.from_numpy(business_context).gt(0).long()
        if self.is_hetero:  # TODO : Convert attributes to Tensors.
            user_attributes = self.attribute_data.user_attributes[user_idx, :]
            business_attributes = self.attribute_data. \
                business_attributes[business_idx, :]
            return (user_idx, business_idx), \
                (prediction, user_context, business_context, user_attributes, business_attributes)
        else:
            return (user_idx, business_idx), \
                (prediction, user_context, business_context)

    def attribute_stats(self):
        n_user_attrs = self.attribute_data.user_attributes.shape[1]
        user_attrs_stats = []
        for idx in range(n_user_attrs):
            data = self.attribute_data.user_attributes[:,idx]
            stats = { 'mean': np.mean(data), 'std': np.std(data) }
            user_attrs_stats.append(stats)

        n_business_attrs = self.attribute_data.business_attributes.shape[1]
        business_attrs_stats = []
        for idx in range(n_business_attrs):
            data = self.attribute_data.business_attributes[:,idx]
            stats = { 'mean': np.mean(data), 'std': np.std(data) }
            business_attrs_stats.append(stats)
        return user_attrs_stats, business_attrs_stats

    def to_PACE_format(self, sample_percent=0.04):
        n_entries = self.poi_user_data.relations_count()
        n_samples = int(n_entries * sample_percent)
        print('number of samples', n_samples)
        u_context = []
        s_context = []
        u_props = []
        s_props = []

        user_input, item_input, ui_label = self._pace_inputs(n_samples)

        b_graph_map = np.unique([self.business_data.business_id2idx[self.poi_user_data.idx2business_id[idx]]
            for idx in item_input])
        u_graph_map = np.unique([self.user_data.user_id2idx[self.poi_user_data.idx2user_id[idx]]
            for idx in user_input])

        u_size = len(u_graph_map)
        s_size = len(b_graph_map)
        print('number of unique users', u_size)
        print('number of unique spots', s_size)

        if self.is_hetero:
            u_props = self.attribute_data.user_props_as_list(user_input)
            s_props = self.attribute_data.business_props_as_list(item_input)

        for idx in range(len(user_input)):
            print('creating context idx=', idx, end='\r')
            if ui_label[idx] == 0:
                uc = np.zeros(u_size)
                sc = np.zeros(s_size)
            else:
                uid = self.poi_user_data.idx2user_id[item_input[idx]]
                if uid in self.user_data.user_id2idx:
                    user_idx = self.user_data.user_id2idx[uid]
                    uc = np.greater_equal(self.user_data.user_edges[user_idx].A[:, u_graph_map].reshape(-1), 1).astype(int)
                else:
                    uc = np.zeros(u_size)

                bid = self.poi_user_data.idx2business_id[item_input[idx]]
                if bid in self.business_data.business_id2idx:
                    business_idx = self.business_data.business_id2idx[bid]
                    sc = self.business_data.business_edges[business_idx].A[:, b_graph_map].reshape(-1)
                else:
                    sc = np.zeros(s_size)

            u_context.append(uc)
            s_context.append(sc)

        return { 'user_input': user_input,
                 'item_input': item_input,
                 'ui_label': ui_label,
                 'u_context': u_context,
                 's_context': s_context,
                 'u_props': u_props,
                 's_props': s_props }

    def _pace_inputs(self, n_samples):
        user_inputs = []
        business_inputs = []
        ui_label = []
        for virtual_relation_idx in range(n_samples):
            relation_idx = (virtual_relation_idx // self.negative_samples) % self.n_relations
            negative_sample = virtual_relation_idx % self.negative_samples
            if not(negative_sample == 0):
                if (np.random.randint(2) == 0):  # Negative sample the user_id
                    business_idx = self.poi_user_data.business_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
                    user_idx = np.random.randint(0, self.n_users)
                    while self.poi_user_data.matrix[user_idx, business_idx] != 0:
                        user_idx = np.random.randint(0, self.n_users)
                else:  # Negative sample the business_id
                    user_idx = self.poi_user_data.user_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
                    business_idx = np.random.randint(0, self.n_businesses)
                    while self.poi_user_data.matrix[user_idx, business_idx] != 0:
                        business_idx = np.random.randint(0, self.n_businesses)
                user_inputs.append(user_idx)
                business_inputs.append(business_idx)
                ui_label.append(0)
            else:  # Get a sample from the data using relation_idx
                user_idx = self.poi_user_data.user_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
                business_idx = self.poi_user_data.business_ids[relation_idx]  # looks scary but is a naming mistake in PoiUserData
                user_inputs.append(user_idx)
                business_inputs.append(business_idx)
                ui_label.append(1)
        return user_inputs, business_inputs, ui_label

    def PACE_sample(self, path_portion=0.01, path_length=10, samples_num=5, window_size=3):
        pass

if __name__ == '__main__':
    # Other arguments are given in as default arguments,
    sanity_test = YelpDataset('./', is_hetero=True)
    all = sanity_test.to_PACE_format()
    import pdb; pdb.set_trace()
    x = {}
    x['user_input'] = all['user_input']
    x['item_input'] = all['item_input']
    x['ui_label'] = all['ui_label']

    c = {}
    c['u_context'] = all['u_context']
    c['s_context'] = all['s_context']

    import pickle
    with open('x.pace.format.pkl', 'wb') as f:
        pickle.dump(x, f)

    with open('c.pace.format.pkl', 'wb') as f:
        pickle.dump(c, f)
    import pdb; pdb.set_trace()
