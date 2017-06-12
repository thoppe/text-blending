import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from sampler import TextSampler
import sys, os
import h5py
import numpy as np

#############################################################
batch_size = 512
n_epochs = 20
n_size = 300
learning_rate = 0.0025
#############################################################

f_h5 = 'master_model/vectorized_sequences.h5'
cutoff = 10**20

with h5py.File(f_h5,'r') as h5:

    X = h5['X'][:cutoff]
    y = h5['y'][:cutoff]

    maxlen = h5.attrs['maxlen']
    chars = h5.attrs['chars']

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


print 'Build model...'
s_in = (maxlen, len(chars))
model = Sequential()
model.add(LSTM(n_size, input_shape=s_in,
               return_sequences=True,activation='softsign'))
model.add(LSTM(n_size,activation='softsign'))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
start_iter = 1
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(lr=learning_rate),
)


#f_load = "saved_models/model_004_1.3162.h5"
#start_iter = int(os.path.basename(f_load).split('_')[1])+1
#model = keras.models.load_model(f_load)

os.system('mkdir -p saved_models/')
f_save = "saved_models/model_{epoch:03d}_{val_loss:.4f}.h5"
checkpointer = ModelCheckpoint(
    filepath=f_save,
    save_weights_only=False,
    period=1,
    verbose=1,
    save_best_only=True
)

model.fit(
    X, y,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=[checkpointer, ],
    initial_epoch=start_iter,
)
