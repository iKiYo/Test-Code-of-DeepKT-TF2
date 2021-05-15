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


class DKTAccum_no_count_Model(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=100, hidden_units=100, dropout_rate=0.2, delta_format="both", delta_dim="single"):   
    x = tf.keras.Input(shape=(None, num_skills*2), name='x')
    seq_delta = tf.keras.Input(shape=(None, 1), name='seq_delta')
    rep_delta = tf.keras.Input(shape=(None, 1), name='rep_delta')
    q = tf.keras.Input(shape=(None, num_skills), name='q')

    count_mask = layers.Masking(mask_value=0, input_shape=(None, 2*num_skills))

    # normal input x
    x_emb = layers.TimeDistributed(layers.Dense(embed_dim))

    # time
    seq_emb = layers.TimeDistributed(layers.Dense(embed_dim)) 
    rep_emb = layers.TimeDistributed(layers.Dense(embed_dim)) 

    # combine x and _c_t
    c_dot = layers.Multiply()
    c_concat = layers.Concatenate()

    mask = layers.Masking(mask_value=0.0, input_shape=(None, embed_dim))
    lstm =  layers.LSTM(hidden_units, return_sequences=True)
    out_dropout =  layers.TimeDistributed(layers.Dropout(dropout_rate))
    out_sigmoid =  layers.TimeDistributed(layers.Dense(num_skills,  activation='sigmoid'))
    dot =  layers.Multiply()
    reduce_sum =  layers.Dense(1, trainable=False, 
                               kernel_initializer=tf.keras.initializers.constant(value=1),
                               name='outputs')

    # define graph
    # normal input x
    embed_x = x_emb(x)
    

    # forgetting curve
    seq_pre_delta = tf.math.log1p(seq_delta)
    rep_pre_delta = tf.math.log1p(rep_delta)
    if delta_dim == "one-hot":
      seq_pre_delta = seq_pre_delta * x 
      rep_pre_delta = rep_pre_delta * x
    if delta_dim == "seq-one-hot":
      seq_pre_delta = seq_pre_delta * x 
    if delta_dim == "rep-one-hot":
      rep_pre_delta = rep_pre_delta * x

    seq_embed_delta=seq_emb(seq_pre_delta)
    rep_embed_delta=rep_emb(rep_pre_delta)
  
    if delta_format == "seq":
      x_c_t = c_concat([embed_x, seq_embed_delta]) # N+N
    elif delta_format == "rep":
      x_c_t = c_concat([embed_x, rep_embed_delta]) # N+N
    elif delta_format == "both":
      x_c_t = c_concat([embed_x, seq_embed_delta, rep_embed_delta]) # N+N
    else:
      raise ValueError("delta format sould be seq, rep or both")

    tempo_mask = count_mask.compute_mask(x)
    h = lstm(x_c_t, mask=tempo_mask)

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    outputs = reduce_sum(y_pred)

    super().__init__(inputs=[x, seq_delta, rep_delta, q], outputs=outputs, name="DKTAccum_no_count_Model")
