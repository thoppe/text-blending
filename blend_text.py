from __future__ import division
import keras
from build_training_set import load_info
from sampler import TextSampler
import sys, os
import collections
import numpy as np
import h5py
import random
from tqdm import tqdm

#############################################################
n_sampled_chars = 750
temperature = 0.85
batch_size = 256//2

#starting_text = "war never changes "
starting_text = "i will have nothing more to do with you "

#f1 = "single_models/shakespeare_input.h5"
#f2 = "single_models/LoveCraft.h5"

#f1 = "single_models/Delphi_Complete_Works_of_Ernest_Hemingway.h5"
#f2 = "single_models/Alice_in_Wonderland.h5"

#f1 = "single_models/The_Complete_Sherlock_Holmes,_Illustrated.h5"
#f2 = "single_models/HarryPotter.h5"

f1 = "single_models/warpeace_input.h5"
f2 = "single_models/LoveCraft.h5"


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

def clean_text(text):
    
    if text[-1] == ' ':
        tokens = text.split()
        last_word = ''.join([x for x in tokens[-1] if x.isalpha()])
        if last_word not in words:
            text = ' '.join(tokens[:-1])
        #else:
        #    print text.strip()
        
    return text

words = read_words(f1)
words.update(read_words(f2))
print "Found {} words".format(len(words))

fix_model(f1); fix_model(f2);


M1 = keras.models.load_model(f1)
M2 = keras.models.load_model(f2)

TS = TextSampler(M1, maxlen, char_indices, indices_char)

text = TS.format_text(starting_text)
TEXT = [text,]*batch_size

#for n in range(n_sampled_chars+1):
ALPHA = np.linspace(-0.05, 1.15, n_sampled_chars)

#for n in tqdm(range(n_sampled_chars)):
progress_bar = tqdm(total=n_sampled_chars*1.2)
while True:

    progress_bar.update()
    N = np.array([len(x.strip())/n_sampled_chars for x in TEXT])
    
    if N.mean() >= 0.9:
        break
    
    X = np.array(map(TS.text_to_vec, map(TS.format_text, TEXT)))
    X = X.reshape([batch_size, maxlen, len(char_indices)])
    p1 = M1.predict(X)
    p2 = M2.predict(X)

    idx1 = np.array([TS.sample(x,temperature) for x in p1])
    idx2 = np.array([TS.sample(x,temperature) for x in p2])

    char1 = np.array([indices_char[i] for i in idx1])
    char2 = np.array([indices_char[i] for i in idx2])

    prob = random.uniform(0,1)
    for i in range(batch_size):
        n = len(TEXT[i].strip())

        # Full, don't add any more
        if n>=len(ALPHA):
            continue

        print ALPHA[n]
        
        if prob > ALPHA[n]:
            TEXT[i] += char1[i]
        else:
            TEXT[i] += char2[i]

    TEXT = map(clean_text, TEXT)

    print TEXT[0].strip()

name1 = os.path.basename(f1).split('.')[0]
name2 = os.path.basename(f2).split('.')[0]
f_save = "sampled_text/{}_TO_{}.txt".format(name1,name2)

os.system('mkdir -p sampled_text')
with open(f_save,'w') as FOUT:

    for text in TEXT:
        if len(text)<int(n_sampled_chars*0.8):
            continue
        print text.strip()
        FOUT.write(text.strip()+'\n')


