import os
import math
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 50 # 50, 100, 200, or 300 per GLOVE 
HIDDEN_DIM = 5
EPSILON = 1e-8
MULTI_HEADS_COUNT = 3
STACKS_COUNT = 4
DROPOUT_RATE = 0.01

BASE_DIR = os.getenv('HOME') + '/Downloads'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
GLOVE_FILE = 'glove.6B.{}d.txt'.format(EMBEDDING_DIM)
PAD = '<PAD>'
UKNOWN = '<UNK>'

class AttentionLayer(keras.engine.topology.Layer):
    def __init__(self, multihead=4, dropout_rate=DROPOUT_RATE, is_training=True, **kwargs):
        super(AttentionLayer, self).__init__()
        self.dropout_rate = dropout_rate 
        self.is_training = is_training
        self.multihead = multihead

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape) 

        weights = []
        for head in range(self.multihead):
            w = []
            for i, n in enumerate(['q', 'k', 'v']):
                name = 'head_{}_{}_weights'.format(head, i)
                if n == 'q':
                    shape = (input_shape[0][-1], HIDDEN_DIM)
                elif n == 'k':
                    shape = (input_shape[1][-1], HIDDEN_DIM)
                else:
                    shape = (input_shape[2][-1], HIDDEN_DIM)
                ww = self.add_weight(name=name, shape=shape, initializer='uniform', trainable=self.is_training)
                w.append(ww)
            weights.append(w)

        self.w = weights
        self.z_weights = self.add_weight(name='z_weights', shape=(HIDDEN_DIM * self.multihead, HIDDEN_DIM), initializer='uniform', trainable=self.is_training)

    def call(self, input, **kwargs):
        queries = input[0]
        keys = input[1]
        values = input[2]

        z = []
        for head in range(self.multihead):
            weights = self.w[head]
            q_weights = weights[0]
            k_weights = weights[1]
            v_weights = weights[2]
            # print(head)
            # print(queries.shape)
            # print(keys.shape)

            # print(q_weights.shape)
            # print(k_weights.shape)
            # print(v_weights.shape)

            q_w = tf.tile(tf.expand_dims(q_weights, 0), (tf.shape(queries)[0], 1, 1))
            Q = tf.matmul(queries, q_w)

            k_w = tf.tile(tf.expand_dims(k_weights, 0), (tf.shape(keys)[0], 1, 1))
            K = tf.matmul(keys, k_w)

            v_w = tf.tile(tf.expand_dims(v_weights, 0), (tf.shape(values)[0], 1, 1))
            V = tf.matmul(values, v_w)

            attention_weights = tf.matmul(Q, K, transpose_b=True)
            attention_weights = attention_weights / (K.get_shape().as_list()[-1] ** 0.5) # Scale

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
            # key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(attention_weights)*(-2**32+1)
            attention_weights = tf.where(tf.equal(key_masks, 0), paddings, attention_weights) # (h*N, T_q, T_k)

            # Causality = Future blinding
            diag_vals = tf.ones_like(attention_weights[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(attention_weights)[0], 1, 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks)*(-2**32+1)
            attention_weights = tf.where(tf.equal(masks, 0), paddings, attention_weights) # (h*N, T_q, T_k)

            softmax_attention_weights = tf.nn.softmax(attention_weights, dim=-1)
            softmax_attention_weights = tf.layers.dropout(softmax_attention_weights, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))

            z_ = tf.matmul(softmax_attention_weights, V)
            z.append(z_)

        z = tf.concat(z, axis=-1)
        z_weights = tf.tile(tf.expand_dims(self.z_weights, 0), (tf.shape(z)[0], 1, 1))
        z = tf.matmul(z, z_weights)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], HIDDEN_DIM)

input = keras.Input(shape=(3,4))

out = AttentionLayer()([input, input, input])
m = keras.Model(input, out)
m.summary()

inp = np.random.randint(0, 10, (2,3,4))
inp[0,0,:] = np.zeros((1, input.shape.as_list()[-1]))
print inp
r = m.predict(inp)
print r