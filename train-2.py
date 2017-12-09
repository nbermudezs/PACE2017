
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
import pickle
from time import time
import dataset

sess = tf.Session()
K.set_session(sess)

def init_normal(shape, name=None):
    return initializers.normal(shape)

def get_Model(num_users, num_items, latent_dim, user_con_len, item_con_len, layers = [20,10,5], regs=[0,0,0]):
	# Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    user_embedding = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer='uniform', W_regularizer = l2(regs[0]), input_length=1)
    item_embedding = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer='uniform', W_regularizer = l2(regs[1]), input_length=1)

    user_latent = Flatten()(user_embedding(user_input))
    item_latent = Flatten()(item_embedding(item_input))

    vector = merge([user_latent, item_latent], mode = 'concat')

    for i in range(len(layers)):
        hidden = Dense(layers[i], activation='relu', init='lecun_uniform', name='ui_hidden_' + str(i))
        vector = hidden(vector)

    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)

    user_context = Dense(user_con_len, activation='sigmoid', init='lecun_uniform', name='user_context')(user_latent)
    item_context = Dense(item_con_len, activation='sigmoid', init='lecun_uniform', name='item_context')(item_latent)

    model = Model(input=[user_input, item_input], output=[prediction, user_context, item_context])
    return model


# In[ ]:


def get_train_instances(train_data):
    while 1:
        user_input = train_data['user_input']
        item_input = train_data['item_input']
        ui_label = train_data['ui_label']
        u_context = train_data['u_context']
        s_context = train_data['s_context']
        for i in range(len(u_context)):
            u = []
            it = []
            p = []
            u.append(user_input[i])
            it.append(item_input[i])
            p.append(ui_label[i])
            x = {'user_input':np.array(u), 'item_input':np.array(it)}
            y = {'prediction':np.array(p), 'user_context':np.array(u_context[i]).reshape((1, u_context_size)), 'item_context':np.array(s_context[i]).reshape((1, s_context_size))}
            yield (x, y)


# In[ ]:

train_data = {}
from Dataset412 import YelpDataset
from UserData import UserData
from PoiUserData import PoiUserData
from BusinessData import BusinessData
from attribute_data import AttributeData
d = YelpDataset('.', is_hetero=True)
train_data = d.to_PACE_format()
u_context = train_data['u_context']
s_context = train_data['s_context']
user_input = train_data['user_input']
item_input = train_data['item_input']
uniq_tmp = np.unique(user_input)
user_input = [np.where(uniq_tmp == x)[0][0] for x in user_input]
uniq_tmp = np.unique(item_input)
item_input = [np.where(uniq_tmp == x)[0][0] for x in item_input]
ui_label = train_data['ui_label']

u_context = np.concatenate((u_context, train_data['u_props']), axis=1)
s_context = np.concatenate((s_context, train_data['s_props']), axis=1)

train_data['u_context'] = u_context
train_data['s_context'] = s_context
train_data['user_input'] = user_input
train_data['item_input'] = item_input

u_context_size = len(u_context[0])
s_context_size = len(s_context[0])

# In[ ]:


model = get_Model(num_users=100000,
                  num_items=100000,
                  latent_dim=10,
                  user_con_len=u_context_size,
                  item_con_len=s_context_size)
config = model.get_config()
weights = model.get_weights()


# In[ ]:

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


if __name__ == '__main__':
    # import pdb; pdb.set_trace()

    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    learner = "Adam"
    learning_rate = 0.0001
    epochs = 100
    batch_size = 1024
    verbose = 1
    losses = ['binary_crossentropy','categorical_crossentropy', 'categorical_crossentropy']

    num_users, num_items = len(user_input), len(item_input)
    import pdb; pdb.set_trace()

    print('Build model')
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accs = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            # print('LOSS: ', logs.get('loss'), logs.get('acc'))
            self.accs.append(logs.get('acc'))

    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)

    history = LossHistory()
    model = get_Model(num_users, num_items, 10, u_context_size, s_context_size, layers, reg_layers)

    model.compile(optimizer=Adam(lr=learning_rate), loss=losses, metrics=['accuracy', precision, recall])


    print('Start Training')

    for epoch in range(epochs):
        t1 = time()
        hist = model.fit_generator(get_train_instances(train_data), samples_per_epoch=batch_size, nb_epoch=10, verbose=1, callbacks=[history,board])
        t2 = time()
        print(epoch, t2-t1)
