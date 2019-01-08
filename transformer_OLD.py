from __future__ import print_function
import os
import math
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 50 # 50, 100, 200, or 300 per GLOVE 
HIDDEN_DIM = 32
EPSILON = 1e-8
MULTI_HEADS_COUNT = 6
STACKS_COUNT = 4
DROPOUT_RATE = 0.01

BASE_DIR = os.getenv('HOME') + '/Downloads'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
GLOVE_FILE = 'glove.6B.{}d.txt'.format(EMBEDDING_DIM)
PAD = '<PAD>'
UKNOWN = '<UNK>'
SENT_START = '<S>'
SENT_END = '</S>'

def residual(input):
    inputs_shape = (MAX_SEQUENCE_LENGTH, HIDDEN_DIM)
    params_shape = (HIDDEN_DIM,)
    
    # mean, variance = tf.nn.moments(input, -1, keep_dims=True)
    mean = keras.backend.mean(input, axis=-1, keepdims=True)
    variance = keras.backend.var(input, axis=-1, keepdims=True)
    # beta= tf.Variable(tf.zeros(params_shape))
    # gamma = tf.Variable(tf.ones(params_shape))
    beta = np.zeros(params_shape)
    gamma = np.ones(params_shape)
    normalized = (input - mean) / ((variance + EPSILON) ** (.5))
    outputs = gamma * normalized + beta
    outputs += input
    # return keras.backend.update_add(input, tf.convert_to_tensor(outputs))
    return outputs

def attention(queries, keys=None, head_count=MULTI_HEADS_COUNT, dropout_rate=DROPOUT_RATE, is_training=True):
    if keys is None:
        keys = queries # self-attention    

    values = []
    for _ in range(head_count):
        Q = keras.layers.Dense(HIDDEN_DIM, activation='relu')(queries)
        K = keras.layers.Dense(HIDDEN_DIM, activation='relu')(keys)
        V = keras.layers.Dense(HIDDEN_DIM, activation='relu')(keys)
        attention_weights = keras.backend.batch_dot(Q, keras.backend.permute_dimensions(K, (0, 2, 1)))
        attention_weights = attention_weights / (HIDDEN_DIM ** 0.5) # Scale
        softmax_attention_weights = keras.backend.softmax(attention_weights)
        softmax_attention_weights = keras.layers.Dropout(dropout_rate)(softmax_attention_weights)
        z_ = keras.backend.batch_dot(softmax_attention_weights, V)
        values.append(z_)

    z = keras.backend.concatenate(values, axis=-1)
    z = keras.layers.Dense(HIDDEN_DIM)(z)
    return z

def positional_encode(input):
    # _, N, T = input.get_shape().as_list()
    N = MAX_SEQUENCE_LENGTH
    T = EMBEDDING_DIM
    pe = np.zeros((N,T))
    position = np.arange(0, N)
    position = np.expand_dims(position, 1)
    div_term = np.exp(np.arange(0, T, 2) * -(math.log(10000.0) / T))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    # pe = tf.cast(tf.convert_to_tensor(pe), dtype=tf.float32)
    return input + pe

def load_glove_data():
    glove_weights = []
    word2idx = {}
    idx2word = {}
    with open(os.path.join(GLOVE_DIR, GLOVE_FILE)) as f:
        # handle PAD
        word2idx[PAD] = 0
        idx2word[0] = PAD
        glove_weights.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))

        # handle UNKNOWN
        word2idx[UKNOWN] = 1
        idx2word[1] = UKNOWN
        glove_weights.append(np.ones(EMBEDDING_DIM, dtype=np.float32))

        # handle SENT_START
        word2idx[SENT_START] = 2
        idx2word[2] = SENT_START
        glove_weights.append(np.ones(EMBEDDING_DIM, dtype=np.float32) + 1)

        # handle SENT_END
        word2idx[SENT_END] = 3
        idx2word[3] = SENT_END
        glove_weights.append(np.ones(EMBEDDING_DIM, dtype=np.float32) + 2)

        for i, line in enumerate(f):
            idx = i + 4
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_weights.append(coefs)
            word2idx[word] = idx
            idx2word[idx] = word

        glove_weights = np.asarray(glove_weights)
    return word2idx, idx2word, glove_weights   

class Transformer():
    def __init__(self, stacks_count=STACKS_COUNT):
        self.word2idx, self.idx2word, self.glove_weights = load_glove_data()    
        # glove_embedding_layer = GloveEmbeddingLayer(EMBEDDING_DIM, self.word2idx, self.idx2word, self.glove_weights)
        embedding_layer = keras.layers.Embedding(
            len(self.word2idx), 
            EMBEDDING_DIM, 
            embeddings_initializer=keras.initializers.Constant(self.glove_weights), 
            input_length=MAX_SEQUENCE_LENGTH, 
            trainable=False)

        # encoder        
        encoder_inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), name='encoder_inputs')
        encoder_embedding = embedding_layer(encoder_inputs)
        # encoder_positional_encoded = keras.layers.Lambda(positional_encode)(encoder_embedding)
        encoder_positional_encoded = positional_encode(encoder_embedding)
        encoder_dropout1 = keras.layers.Dropout(DROPOUT_RATE)(encoder_positional_encoded)        
        encoder_in_ = encoder_dropout1
        for _ in range(stacks_count):
            encoder_out_ = self.encoder(encoder_in_)
            encoder_in_ = encoder_out_

        # decoder
        decoder_inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), name='decoder_inputs')
        decoder_embedding = embedding_layer(decoder_inputs)
        # decoder_positional_encoded = keras.layers.Lambda(positional_encode)(decoder_embedding)
        decoder_positional_encoded = positional_encode(decoder_embedding)
        decoder_dropout1 = keras.layers.Dropout(DROPOUT_RATE)(decoder_positional_encoded) 
        decoder_in_ = decoder_dropout1
        for _ in range(stacks_count):
            decoder_out_ = self.decoder(decoder_in_, encoder_out_)
            decoder_in_ = decoder_out_

        preds = keras.layers.Dense(len(self.word2idx), activation='softmax')(decoder_out_)

        # self.model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=encoder_out_)
        self.model = keras.Model(inputs=encoder_inputs, outputs=encoder_out_)

        # TODO 
        # - summary does not reflect number of params per custom layer
        # - https://stackoverflow.com/questions/46519024/keras-unable-to-calculate-number-of-parameters-in-a-keras-custom-layer
        print(self.model.summary())

    def encoder(self, inputs):
        # self_attention_out = MultiheadAttentionLayer(queries=inputs)(inputs) # self attention
        self_attention_out = attention(queries=inputs, keys=None)
        residual_out1 = keras.layers.Lambda(residual)(self_attention_out)
        feedforward_out = keras.layers.Dense(HIDDEN_DIM)(residual_out1) # feedforward layer
        return residual(feedforward_out)

    def decoder(self, inputs, encoder_inputs):
        self_attention_out = attention(queries=inputs, keys=inputs)
        residual_out1 = residual(self_attention_out)
        attention_out = attention(queries=residual_out1, keys=encoder_inputs)
        residual_out2 = residual(self_attention_out)
        feedforward_out = keras.layers.Dense(HIDDEN_DIM)(residual_out2) # feedforward layer
        return residual(feedforward_out)

    def vectorize_texts(self, texts):
        sequences = []
        for text in texts:
            seq = []
            tokens = text_to_word_sequence(text)
            tokens.insert(0, SENT_START)
            tokens.append(SENT_END)
            for t in tokens:
                if t in self.word2idx:
                    seq.append(self.word2idx[t])
                else:
                    seq.append(self.word2idx[UKNOWN])
            sequences.append(seq)
        return pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    def reverse_vectorize(self, vectors):
        texts = []
        for v in vectors:
            text = []
            for idx in v:
                text.append(self.idx2word[idx])
            texts.append(text)
        return texts

    def predict(self, texts):
        encoder_inputs = self.vectorize_texts(texts)
        decoder_inputs = np.zeros((len(texts), MAX_SEQUENCE_LENGTH), dtype=np.int32)
        decoder_inputs[:,0] = self.word2idx[SENT_START]

        for i in range(MAX_SEQUENCE_LENGTH)[1:]:
            result = self.model.predict([encoder_inputs, decoder_inputs])
            decoder_inputs[:,i] = np.argmax(result[:, i, :], axis=1)

        return self.reverse_vectorize(decoder_inputs)

if __name__ == '__main__':
    t = Transformer()
    encoder_texts = [
        'xyop hello hayward blah baz xkja austin chau my friend! how are you man?!', 
        'oh this is going to be so good'
    ]
    decoder_texts = [
        'xyop hello hayward blah baz xkja austin chau my friend! how are you man?!', 
        'oh this is going to be so good'
    ]

    # results = t.predict(encoder_texts)
    # print(results)