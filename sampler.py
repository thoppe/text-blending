import keras
import numpy as np
import sys

class TextSampler(keras.callbacks.Callback):

    def __init__(self,
                 model,
                 maxlen, char_indicies, indices_char,
                 *args, **kwargs):
        self.maxlen = maxlen
        self.char_indices = char_indicies
        self.indices_char = indices_char
        self.model = model
        self.starter_text = "battle waged on, but her emails, "
        #keras.callbacks.Callback.__init__(*args, **kwargs)
    
    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        self.get_text(T=0.5)
        self.get_text(T=1.0)
    
    def sample(self, preds, temperature=1.0):
        epsilon = 10.0**-5
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds+epsilon) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def format_text(self, text):
        while len(text) > self.maxlen:
            text = text[1:]
        while len(text) < self.maxlen:
            text = ' ' + text
        return text
    
    def text_to_vec(self, text):
        x = np.zeros((1, self.maxlen, len(self.char_indices)))
        for t, char in enumerate(text):
            x[0, t, self.char_indices[char]] = 1.
        return x

    def get_text(self, T, sequence_length=200):
        
        sentence = self.format_text(self.starter_text)
        generated = sentence
        print
        sys.stdout.write(generated)

        for i in range(sequence_length):
            x = self.text_to_vec(sentence)
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, T)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
