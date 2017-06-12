from __future__ import division
import keras
from build_training_set import load_info
from sampler import TextSampler
import sys, os
import collections
import numpy as np
import h5py
#############################################################
n_sampled_chars = 1000
temperature = 0.85
starting_text = "war never changes "
#starting_text = ". mankind "

f1 = "single_models/warpeace_input.h5"
f2 = "single_models/Delphi_Complete_Works_of_Ernest_Hemingway.h5"
#############################################################

f_vectors = 'master_model/vectorized_sequences.h5'
maxlen, chars, char_indices, indices_char = load_info(f_vectors)

def fix_model(f_h5):
    # https://github.com/fchollet/keras/issues/4044
    with h5py.File(f_h5,'r+') as h5:
        if "optimizer_weights" in h5:
            del h5["optimizer_weights"]

def read_words(f_h5):
    f_txt = os.path.join('clean_txt/', os.path.basename(f_h5))
    f_txt = f_txt.replace('.h5', '.txt')
    with open(f_txt) as FIN:
        text = FIN.read().lower()
    tokens = set(text.split())
    tokens = [''.join([c for c in x if c.isalpha()]) for x in tokens]
    tokens = set(tokens)
    return tokens

words = read_words(f1)
words.update(read_words(f2))
print "Found {} words".format(len(words))

fix_model(f1); fix_model(f2);


M1 = keras.models.load_model(f1)
M2 = keras.models.load_model(f2)

TS = TextSampler(M1, maxlen, char_indices, indices_char)

text = TS.format_text(starting_text)

#for n in range(n_sampled_chars+1):
for alpha in np.linspace(-0.25, 1.25, n_sampled_chars):

    x = TS.text_to_vec(TS.format_text(text))

    
    if alpha < 0.5:
        p = M1.predict(x)[0]
        idx = TS.sample(p, temperature)
        char = indices_char[idx]
    else:
        p = M2.predict(x)[0]
        idx = TS.sample(p, temperature)
        char = indices_char[idx]
    
    '''
    alpha = np.clip([alpha],0,1)[0]
    p1 = M1.predict(x)[0]*(1-alpha)
    p2 = M2.predict(x)[0]*(alpha)
    p = p1+p2
    p /= p.sum()
    idx = TS.sample(p, temperature)
    char = indices_char[idx]
    '''
    

    text += char

    if char == ' ':
        tokens = text.split()
        last_word = ''.join([x for x in tokens[-1] if x.isalpha()])
        if last_word not in words:
            text = ' '.join(tokens[:-1])
        else:
            print text.strip()
