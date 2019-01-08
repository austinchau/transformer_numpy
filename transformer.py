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
MULTI_HEADS_COUNT = 3
STACKS_COUNT = 2
DROPOUT_RATE = 0.01

BASE_DIR = os.getenv('HOME') + '/Downloads'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
GLOVE_FILE = 'glove.6B.{}d.txt'.format(EMBEDDING_DIM)
PAD = '<PAD>'
UKNOWN = '<UNK>'
SENT_START = '<S>'
SENT_END = '</S>'

class ResidualLayer(keras.engine.topology.Layer):
    def __init__(self):
        super(ResidualLayer, self).__init__()

    def build(self, input_shape):
        super(ResidualLayer, self).build(input_shape)   

    def call(self, input):
        inputs_shape = input.get_shape()
        params_shape = inputs_shape[-1:]
        
        mean, variance = tf.nn.moments(input, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (input - mean) / ((variance + EPSILON) ** (.5))
        outputs = gamma * normalized + beta
        outputs += input
        return outputs

class AttentionLayer(keras.engine.topology.Layer):
    def __init__(self, future_blinding=False, multi_head=MULTI_HEADS_COUNT, dropout_rate=DROPOUT_RATE, is_training=True, **kwargs):
        super(AttentionLayer, self).__init__()
        self.dropout_rate = dropout_rate 
        self.is_training = is_training
        self.multi_head = multi_head
        self.future_blinding = future_blinding

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape) 

        weights = []
        for head in range(self.multi_head):
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
        self.z_weights = self.add_weight(name='z_weights', shape=(HIDDEN_DIM * self.multi_head, HIDDEN_DIM), initializer='uniform', trainable=self.is_training)

    def call(self, input, **kwargs):
        queries = input[0]
        keys = input[1]
        values = input[2]

        z = []
        for head in range(self.multi_head):
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

            if self.future_blinding:
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

class PositionalEncodingLayer(keras.engine.topology.Layer):
    def __init__(self):
        super(PositionalEncodingLayer, self).__init__()

    def build(self, input_shape):
        super(PositionalEncodingLayer, self).build(input_shape)

    def call(self, input):
        _, N, T = input.get_shape().as_list()
        pe = np.zeros((N,T))
        position = np.arange(0, N)
        position = np.expand_dims(position, 1)
        div_term = np.exp(np.arange(0, T, 2) * -(math.log(10000.0) / T))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = tf.cast(tf.convert_to_tensor(pe), dtype=tf.float32)
        return input + pe

class GloveEmbeddingLayer(keras.engine.topology.Layer):
    def __init__(self, embedding_dim, word2idx, idx2word, glove_weights):
        super(GloveEmbeddingLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.word2idx, self.idx2word, self.glove_weights = word2idx, idx2word, glove_weights

    def build(self, input_shape):
        super(GloveEmbeddingLayer, self).build(input_shape)
        with tf.variable_scope('glove_embedding', reuse=tf.AUTO_REUSE):
            self.embedding_weights = tf.get_variable(
                name='embedding_weights', 
                shape=(len(self.glove_weights), self.embedding_dim),
                initializer=tf.constant_initializer(self.glove_weights),
                trainable=False)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)

    def call(self, input):
        input = tf.cast(input, tf.int32)
        return tf.nn.embedding_lookup(self.embedding_weights, input)  


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
        glove_embedding_layer = GloveEmbeddingLayer(EMBEDDING_DIM, self.word2idx, self.idx2word, self.glove_weights)

        # encoder        
        encoder_inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), name='encoder_inputs')
        encoder_embedding = glove_embedding_layer(encoder_inputs)
        encoder_positional_encoded = PositionalEncodingLayer()(encoder_embedding)
        encoder_dropout1 = keras.layers.Dropout(DROPOUT_RATE)(encoder_positional_encoded)   
        encoder_out_ = self.encoder(encoder_dropout1)
        encoder_in_ = encoder_dropout1
        for _ in range(stacks_count):
            encoder_out_ = self.encoder(encoder_in_)
            encoder_in_ = encoder_out_

        # decoder
        last_encoder_outputs = encoder_out_
        decoder_inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,), name='decoder_inputs')
        decoder_embedding = glove_embedding_layer(decoder_inputs)
        decoder_positional_encoded = PositionalEncodingLayer()(decoder_embedding)
        decoder_dropout1 = keras.layers.Dropout(DROPOUT_RATE)(decoder_positional_encoded) 
        decoder_out_ = self.decoder(decoder_dropout1, encoder_out_)
        decoder_in_ = decoder_dropout1
        for _ in range(stacks_count):
            decoder_out_ = self.decoder(decoder_in_, last_encoder_outputs)
            decoder_in_ = decoder_out_

        preds = keras.layers.Dense(len(self.word2idx), activation='softmax')(decoder_out_)

        self.model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=preds)

        # TODO 
        # - summary does not reflect number of params per custom layer
        # - https://stackoverflow.com/questions/46519024/keras-unable-to-calculate-number-of-parameters-in-a-keras-custom-layer
        print(self.model.summary())

    def encoder(self, inputs):
        self_attention_out = AttentionLayer()([inputs, inputs, inputs]) # self attention
        residual_out1 = ResidualLayer()(self_attention_out)
        feedforward_out = keras.layers.Dense(HIDDEN_DIM)(residual_out1) # feedforward layer
        return ResidualLayer()(feedforward_out)

    def decoder(self, inputs, encoder_inputs):
        self_attention_out = AttentionLayer(future_blinding=True)([inputs, inputs, inputs]) # self attention
        residual_out1 = ResidualLayer()(self_attention_out)
        plain_attention_out = AttentionLayer()([residual_out1, encoder_inputs, residual_out1]) # plain attention
        residual_out2 = ResidualLayer()(plain_attention_out)
        feedforward_out = keras.layers.Dense(HIDDEN_DIM)(residual_out2) # feedforward layer
        return ResidualLayer()(feedforward_out)

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
            next_word = np.argmax(result[:, i, :], axis=1)
            if next_word == SENT_END:
                decoder_inputs[:,i] = next_word
                break
            decoder_inputs[:,i] = next_word

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

    results = t.predict(encoder_texts)
    print(results)