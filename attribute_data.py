'''
    File containing a class for loading and saving
    attributes for users.
    It uses poi-user-data to figure out which users to care for.
'''
import sys
import pdb
import numpy as np
import pickle
import json
from datetime import datetime
from UserData import UserData
from PoiUserData import PoiUserData
from BusinessData import BusinessData
from collections import defaultdict


class AttributeData:
    def __init__(self,
                 poi_user_data_file='poi-users-filtered-09112017.pkl'):
        '''
            Class to wrap around attributes of users and businesses.
            Uses the same indexing as poi_user_data

            relevant members
            ----
                user_attributes
                business_attributes
        '''

        # Get a list of user_ids
        with open(poi_user_data_file, 'rb') as f:
            self.pud = pickle.load(f, encoding='latin1')
        self.uids = set()
        for k, v in self.pud.idx2user_id.items():
            self.uids.add(v)
        self.bids = set()
        for k, v in self.pud.idx2business_id.items():
            self.bids.add(v)
        # Load data from json file and filter it out
        # pdb.set_trace()

    def prepare_user(self, filename='dataset/user.json'):
        self.user_attribute_list = ['review_count', 'average_stars']
        self.user_attributes = np.zeros((len(self.uids),
                                         len(self.user_attribute_list) + 1),
                                        dtype=float)
        count = 0
        with open(filename, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if obj['user_id'] in self.uids:
                    count += 1
                    uidx = self.pud.user_id2idx[obj['user_id']]
                    for attribute_idx in range(len(self.user_attribute_list)):
                        self.user_attributes[uidx, attribute_idx] = \
                            obj[self.user_attribute_list[attribute_idx]]
                    nyears = 2019 - datetime.strptime(obj['yelping_since'],
                                                      '%Y-%M-%d').year
                    # The exact denominator doesn't matter as long
                    # as long as the differential is okay.
                    self.user_attributes[uidx, -1] = len(obj['elite']) / nyears
                else:
                    continue
        print('Got attributes of ', count, ' users')

    def prepare_business(self, filename='dataset/business.json'):
        self.business_attribute_list = ['stars', 'review_count']
        self.business_attributes = np.zeros((len(self.bids),
                                             len(self.business_attribute_list)
                                             ),
                                            dtype=float)
        count = 0
        with open(filename, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if obj['business_id'] in self.bids:
                    count += 1
                    bidx = self.pud.business_id2idx[obj['business_id']]
                    for attrib_idx in range(len(self.business_attribute_list)):
                        self.business_attributes[bidx, attrib_idx] = \
                            obj[self.business_attribute_list[attrib_idx]]
                else:
                    continue
        print('Got attributes for ', count, ' businesses')

if __name__ == '__main__':
    data = AttributeData('poi-users-filtered-09112017.pkl')
    data.prepare_user()
    data.prepare_business()
    with open('attributes0000.11.22.2017.pkl', 'wb') as f:
        pickle.dump(data, f)
    pdb.set_trace()
