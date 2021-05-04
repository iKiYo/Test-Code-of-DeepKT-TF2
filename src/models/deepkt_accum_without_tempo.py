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

  def get_config(self):
    config = super(CountStateRNNCell, self).get_config()
    config.update({"units": self.units})

    return config


class DKTAccum_no_tempo_Model(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=100, hidden_units=100, dropout_rate=0.2, count_format="binary", count_dim="single"):   
    x = tf.keras.Input(shape=(None, num_skills*2), name='x')
    delta = tf.keras.Input(shape=(None, 1), name='delta')
    q = tf.keras.Input(shape=(None, num_skills), name='q')

    # normal input x
    x_emb = layers.TimeDistributed(layers.Dense(embed_dim))

    # count
    count_mask = layers.Masking(mask_value=0, input_shape=(None, 2*num_skills))
    count_cell = CountStateRNNCell(num_skills*2)
    c_count = layers.RNN(count_cell, return_sequences=True)
    c_emb =  layers.TimeDistributed(layers.Dense(embed_dim))
    # c_emb =  layers.TimeDistributed(layers.Dense(2,
    #                                              activation='relu',
    # )
    # )
    # #                                              kernel_constraint=tf.keras.constraints.NonNeg(),
    #                                              kernel_initializer=tf.keras.initializers.RandomUniform(
    # minval=0.05, maxval=2-0.05),
    #                                              bias_constraint=tf.keras.constraints.NonNeg(),
    #                                              bias_initializer =tf.keras.initializers.RandomUniform(
    # minval=0.05, maxval=2-0.05),
    #                                               )
    # )

    # combine x and _c_t
    c_dot = layers.Multiply()
    c_concat = layers.Concatenate()

    mask = layers.Masking(mask_value=0.0, input_shape=(None, embed_dim+ num_skills*2))
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
    
    # count data
    count_t =count_mask(x) # 2M
    # count of all skill/exercises
    count_t = c_count(count_t) # accmulates

    # learning curve 
    count_t = tf.math.log1p(count_t)

    # pick single target skill id of correct/incorrect counts
    index = tf.argmax(x, axis=-1) // 2
    one_hot_correct =tf.one_hot(index*2, num_skills*2)
    one_hot_incorrect = tf.one_hot(index*2+1, num_skills*2) 
    id_tensor = one_hot_correct + one_hot_incorrect
    if count_dim == "single":
      # count features total/correct/incorrect tensor
      correct_count = tf.expand_dims(tf.reduce_sum(one_hot_correct * count_t, axis=-1), axis=-1)
      incorrect_count = tf.expand_dims(tf.reduce_sum(one_hot_incorrect * count_t, axis=-1), axis=-1)
      total_count = tf.expand_dims(tf.reduce_sum(id_tensor * count_t, axis=-1), axis=-1)
      if count_format == "unary": # total number of attempts of target skill
            count_feat = total_count
      elif count_format == "bianry": # separated total correct/incorrect attempts of target skill
        binary_count = c_concat([correct_count, incorrect_count])
        count_feat = binary_count
      else:
        all_count = tf.concat([correct_count, incorrect_count, total_count], axis=-1)
        count_feat = all_count

    else:
      # one hot correct/incorrect tesnor
      if count_format == "unary": # total number of attempts of target skill
        total_count = tf.expand_dims(tf.reduce_sum(id_tensor * count_t, axis=-1), axis=-1)
        id_tensor = tf.one_hot(index, num_skills)
        total_count = tf.tile(total_count, tf.constant([1,1,num_skills], tf.int32))
        one_hot_all_count = total_count * id_tensor 
        count_feat = one_hot_all_count
      else: # binary, separated total correct/incorrect attempts of target skill
        one_hot_binary_count = id_tensor * count_t
        count_feat = one_hot_binary_count

      count_feat = c_emb(count_feat)
  
    # learning curve 
    # logarithm
    # lr_count = tf.math.log1p(count_feat)
    # exponential
    # embed_count = c_emb(binary_count) # 2M to N-1
    # lr_count= tf.ones(shape=(tf.shape(embed_count))) - tf.math.exp(-embed_count)



    # x_c_t = c_concat([embed_x, count_t]) # N+2K
    # x_c_t = c_concat([embed_x, embed_count]) # N+N
    # x_c_t = c_emb(count_t) # 2M to N
    # x_c_t = c_concat([embed_x, one_hot_count]) # N+N
    # no one-hot encoding for counts
    # x_c_t = c_concat([embed_x, correct_count, incorrect_count]) # N+2
    # x_c_t = c_concat([embed_x, total_count]) # N+2
    # x_c_t = c_concat([embed_x, all_count_tensor]) # N+3

    # x_c_t = c_concat([embed_x, lr_count]) # N+2
    x_c_t = c_concat([embed_x, count_feat]) # N+2

    tempo_mask = count_mask.compute_mask(x)
    h = lstm(x_c_t, mask=tempo_mask)

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    outputs = reduce_sum(y_pred)

    super().__init__(inputs=[x, delta, q], outputs=outputs, name="DKTAccumModel")
