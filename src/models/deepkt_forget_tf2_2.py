import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# without hot-enconding for delta_t, no adding of t+1 info to h
# only concatenation
# use x 
class CountStateRNNCell(layers.Layer):

  def __init__(self, units,**kwargs):
      self.units = units
      self.state_size = units
      super(CountStateRNNCell, self).__init__(**kwargs)

  def call(self, inputs, states):
    prev_c_state = states[0]
    output = tf.math.multiply(tf.math.reduce_sum(inputs, axis=1, keepdims=True), prev_c_state)+ inputs

    return output, [output]


class DKTtempoModel_with_x(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=100, hidden_units=100, dropout_rate=0.0):   
    x = tf.keras.Input(shape=(max_sequence_length, num_skills*2), name='x')
    delta = tf.keras.Input(shape=(max_sequence_length, 1), name='delta')
    q = tf.keras.Input(shape=(max_sequence_length, num_skills), name='q')

    count_cell = CountStateRNNCell(num_skills*2)
    c_count = layers.RNN(count_cell, return_sequences=True)

    c_emb = layers.TimeDistributed(
        layers.Dense(embed_dim-1, trainable=True,  #  N-1
                                                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=77),
                                                      input_shape=(None, max_sequence_length, 2*num_skills)) # M to N-1
    )
    
    c_dot = layers.Multiply()
    c_concat = layers.Concatenate()
 
    mask = layers.Masking(mask_value=0.0, input_shape=(max_sequence_length, embed_dim))
    delta_mask = layers.Masking(mask_value=0.0, input_shape=(max_sequence_length,1))
    count_mask = layers.Masking(mask_value=0, input_shape=(max_sequence_length, 2*num_skills))

    lstm =  layers.LSTM(hidden_units, return_sequences=True)
    out_dropout =  layers.TimeDistributed(layers.Dropout(dropout_rate))
    out_sigmoid =  layers.TimeDistributed(layers.Dense(num_skills,  activation='sigmoid'))
    dot =  layers.Multiply()
    # HACK: the shape of q does not fit to Timedistributed operation(may be correct?)
    # dot =  layers.TimeDistributed(layers.Multiply())

    reduce_sum =  layers.Dense(1, trainable=False, 
                                                      kernel_initializer=tf.keras.initializers.constant(value=1),
                                                      input_shape=(None, max_sequence_length,num_skills))
    # reshape layer does not work as graph  # reshape_l = layers.Reshape((-1,6),dynamic=False)#, 
    final_mask =   layers.TimeDistributed(layers.Masking(mask_value=0, 
                                                                                  input_shape=(None, max_sequence_length,1)),
                                                                                  name='outputs'
                                                                                  )

    # define graph
    delta_t =delta_mask(delta) # 1 dimension

    count_t =count_mask(x) # 2M
    count_t = c_count(count_t) # accmulates

    embed_count = c_emb(count_t) # 2M to N-1
    c_t = c_concat([embed_count, delta_t]) # (N-1) + 1 = N

    masked_c_t = mask(c_t)
    h = lstm(masked_c_t)

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    # HACK: without using layer(tf.reduce) might be faster
    # y_pred = reduce_sum(y_pred, axis=2)
    y_pred = reduce_sum(y_pred)
    outputs = final_mask(y_pred)

    super().__init__(inputs=[x, delta, q], outputs=outputs, name="DKTtempoModel_without_x")
