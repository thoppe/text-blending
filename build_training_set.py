import numpy as np
import glob
from tqdm import tqdm
import h5py
import os


def load_all_text(pattern='*'):
    F_TXT = glob.glob("clean_txt/{}.txt".format(pattern))

    TEXT = {}
    for f in F_TXT:
        text = []
        with open(f) as FIN:
            sents = ' '.join(FIN.read().split('\n')).lower()
            text.append(sents)
        TEXT[os.path.basename(f)] = ' '.join(text)
    return TEXT

def build_indicies(text):
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return chars, char_indices, indices_char

def build_sequences(maxlen, step, text):    
    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    return sentences, next_chars


def vectorization(sentences, maxlen, chars,
                  char_indices, next_chars,
                  verbose=False):

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    ITR = enumerate(sentences)
    if verbose: ITR = tqdm(ITR, total=len(sentences))
    for i, sentence in ITR:
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return X, y

def build_master_dataset(maxlen, step, TEXT, f_h5):
    # Builds the master dataset for all others
    
    chars, char_indices, indices_char = build_indicies(TEXT)
    sentences, next_chars = build_sequences(maxlen, step, TEXT)
    X, y = vectorization(sentences, maxlen, chars, char_indices, next_chars)

    with h5py.File(f_h5,'w') as h5:
        h5.attrs['maxlen'] = maxlen
        h5.attrs['chars'] = chars
        h5['X'] = X
        h5['y'] = y

def load_info(f_h5):
    with h5py.File(f_h5,'r') as h5:
        maxlen = h5.attrs['maxlen']
        chars = h5.attrs['chars']
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return maxlen, chars, char_indices, indices_char

def build_single_dataset(text, step, f_h5):
    maxlen, chars, char_indices, indices_char = load_info(f_h5)
    sentences, next_chars = build_sequences(maxlen, step, text)
    X, y = vectorization(sentences, maxlen, chars, char_indices, next_chars)
    return X, y

if __name__ == "__main__":
    TEXT = load_all_text()
    
    f_h5 = 'master_model/vectorized_sequences.h5'
    concat_text = ' '.join(TEXT.values()[:2])
    build_master_dataset(maxlen=60, step=15,
                         TEXT=concat_text, f_h5=f_h5)

    #for key in TEXT:
    #    text = TEXT[key]
    #    print build_single_dataset(text, step=2, f_h5=f_h5)[0].shape
