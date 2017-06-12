import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import os
import collections
from build_training_set import load_all_text, build_single_dataset
from build_training_set import load_info
from sampler import TextSampler

n_epochs = 5
batch_size = 512

TEXT = load_all_text('*')
ordered_keys = sorted(TEXT, key=lambda k: len(TEXT[k]))

f_h5 = 'master_model/vectorized_sequences.h5'
f_master = 'master_model/model_006_1.4274.h5'
f_vectors = 'master_model/vectorized_sequences.h5'
maxlen, chars, char_indices, indices_char = load_info(f_vectors)

os.system('mkdir -p single_models')

#for key in TEXT:
for key in ordered_keys:
    name = '.'.join(key.split('.')[:-1])
    f_save = "single_models/{}.h5".format(name)

    if os.path.exists(f_save):
        print "Already trained", f_save
        continue

    print "Training", key

    text = TEXT[key]
    X, y = build_single_dataset(text, 1, f_h5)

    # Load the master model
    model = keras.models.load_model(f_master)

    # Fix the top layer
    model.layers[0].trainable = False

    TS = TextSampler(model,maxlen,char_indices,indices_char)
    TS.starter_text = text[:maxlen]

    model.fit(
        X, y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=n_epochs,
        #callbacks=[checkpointer,],
    )

    model.save(f_save)
    TS.get_text(1.0, sequence_length=300)
    print

        
