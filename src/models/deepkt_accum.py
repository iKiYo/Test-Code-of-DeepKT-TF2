import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CountStateRNNCell(layers.Layer):

  def __init__(self, units,**kwargs):
      self.units = units
      self.state_size = units
      super(CountStateRNNCell, self).__init__(**kwargs)

  def call(self, inputs, states):
    prev_c_state = states[0]
    output = tf.math.multiply(tf.math.reduce_sum(inputs, axis=1, keepdims=True), prev_c_state)+ inputs

    return output, [output]


class TempoStateRNNCell(layers.Layer):

  def __init__(self, units,**kwargs):
      self.units = units
      self.state_size = units
      super(TempoStateRNNCell, self).__init__(**kwargs)

  # def build(self, input_shape):  # Embedding weights
  #   self.w = tf.Variable(
  #       initial_value=tf.zeros(shape=(input_shape[-1], self.units),
  #                            dtype='float32'),
  #       trainable=False)

  def call(self, inputs, states):
    prev_c_state = states[0]
    # neg_skill(1, 0, 1) dot prev_stat + (delta_t, delta_t, delta_t)
    output = tf.math.multiply(inputs[:,:self.units], prev_c_state[:,:] )  + inputs[:,self.units:]
    # delta_t = tf.math.multiply(inputs[:,:self.units], prev_c_state[:,:self.units])  + inputs[:,self.units:2*self.units]
    # freq = prev_c_state[:,self.units:] + inputs[:,2*self.units:]
    # output = tf.concat([delta_t, freq], 0)
    return output, [output]



class DKTtempoModel_2RNN(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, init_tempo_tensor,
                            embed_dim=100, hidden_units=100, dropout_rate=0.2
                            ):  

    x = tf.keras.Input(shape=(None, num_skills*2), name='x')
    neg_sk = tf.keras.Input(shape=(None, num_skills), name='neg_skill')
    delta = tf.keras.Input(shape=(None, num_skills), name='delta')
    q = tf.keras.Input(shape=(None, num_skills), name='q')

    tempo_cell = TempoStateRNNCell(num_skills)
    rnn_tempo = layers.RNN(tempo_cell, return_sequences=True)
    count_cell = CountStateRNNCell(num_skills*2)
    rnn_count = layers.RNN(count_cell, return_sequences=True)



    forget_dense_1 =  layers.TimeDistributed(
        layers.Dense(num_skills, trainable=True, activation='sigmoid',
                    input_shape=(None, None, num_skills),
                     name='forget_dense_1')
    )
    forget_dense_2 =  layers.TimeDistributed(
        layers.Dense(num_skills, trainable=True, 
                     input_shape=(None, None, num_skills),
                     name='forget_dense_2')
    )
    forget_dense_1_2 =  layers.TimeDistributed(
        layers.Dense(num_skills, trainable=True, 
                     input_shape=(None, None, num_skills),
                     name='forget_dense_1_2')
    )
    forget_dense_2_2 =  layers.TimeDistributed(
        layers.Dense(num_skills, trainable=True, 
                     input_shape=(None, None, num_skills),
                     name='forget_dense_2_2')
)
    
    count_emb = layers.TimeDistributed(
        layers.Dense(num_skills, trainable=True, 
                     input_shape=(None, None, num_skills*2))
)
    learn_dense = layers.TimeDistributed(
        layers.Dense(num_skills, trainable=True, 
                     input_shape=(None, None, num_skills),
                     name='learn_dense')
    )
    c_emb = layers.TimeDistributed(
        layers.Dense(embed_dim, trainable=True, 
                     input_shape=(None, None, num_skills*2))
    )
    
    x_emb = layers.TimeDistributed(
        layers.Dense(embed_dim, trainable=True,  
                     input_shape=(None, None, 2*num_skills)) # N
    )
    
    concat = layers.Concatenate()

    m_mask = layers.Masking(mask_value=0, input_shape=(None, num_skills))
    m2_mask = layers.Masking(mask_value=0, input_shape=(None, num_skills*2))
    mask = layers.Masking(mask_value=0.0, input_shape=(None, embed_dim+num_skills))

    lstm =  layers.LSTM(hidden_units, return_sequences=True)
    out_dropout =  layers.TimeDistributed(layers.Dropout(dropout_rate))
    out_sigmoid =  layers.TimeDistributed(layers.Dense(num_skills,  activation='sigmoid'))
    c_dot =  layers.Multiply()
    dot =  layers.Multiply()
    # HACK: the shape of q does not fit to Timedistributed operation(may be correct?)
    reduce_sum =  layers.Dense(1, trainable=False,
                               kernel_initializer=\
                               tf.keras.initializers.constant(value=1),
                               input_shape=(None, None,num_skills),
                               name='outputs')
    

    # define graph
    delta_t = concat([neg_sk, delta]) # 2M
    delta_t = m2_mask(delta_t)
    delta_t = rnn_tempo(delta_t, initial_state=init_tempo_tensor) # 2M to M
  
    delta_t = forget_dense_1(delta_t)
    delta_t = tf.math.exp(-delta_t)
    delta_t = forget_dense_2(delta_t) # M

    count_t =m2_mask(x) # 2M
    count_t = rnn_count(count_t) # accmulates
    embed_count = count_emb(count_t) # 2M to M
    dotted_c_t = c_dot([delta_t, embed_count]) #M
    
    embed_x = x_emb(x)

    x_c_t = concat([embed_x, dotted_c_t, delta_t]) # N+M + M

    x_mask = m2_mask.compute_mask(x)
    h = lstm(x_c_t, mask=x_mask)

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    outputs = reduce_sum(y_pred)

    super().__init__(inputs=[x, neg_sk, delta, q], outputs=outputs, name="DKTtempoModel_2RNN")

