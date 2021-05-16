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


class DKTAccumModel(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=100, hidden_units=100, dropout_rate=0.2):   
    x = tf.keras.Input(shape=(None, num_skills*2), name='x')
    # delta = tf.keras.Input(shape=(None, 1), name='delta')
    seq_delta = tf.keras.Input(shape=(None, 1), name='seq_delta')
    rep_delta = tf.keras.Input(shape=(None, 1), name='rep_delta')
    q = tf.keras.Input(shape=(None, num_skills), name='q')

    # normal input x
    x_emb = layers.TimeDistributed(layers.Dense(embed_dim))

    # count
    count_mask = layers.Masking(mask_value=0, input_shape=(None, 2*num_skills))
    count_cell = CountStateRNNCell(num_skills*2)
    c_count = layers.RNN(count_cell, return_sequences=True)
    c_emb =  layers.TimeDistributed(layers.Dense(embed_dim))
#     c_emb =  layers.TimeDistributed(layers.Dense(num_skills*2, activation='relu',
#     kernel_constraint=tf.keras.constraints.MinMaxNorm(
#     min_value=0.0, max_value=1.0, rate=1.0, axis=0
# )))
# tf.keras.constraints.NonNeg()))
    # c_coef =  layers.TimeDistributed(layers.Dense(embed_dim))
    # time
    seq_emb = layers.TimeDistributed(layers.Dense(embed_dim)) 
    rep_emb = layers.TimeDistributed(layers.Dense(embed_dim)) 
    # delta_emb = layers.TimeDistributed(layers.Dense(1))
    # delta_coeff = layers.TimeDistributed(layers.Dense(1))

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
    
    # count data
    count_t =count_mask(x) # 2M
    count_t = c_count(count_t) # accmulates

    # *** learning curve method 
    # pick single target skill id of correct/incorrect counts
    index = tf.argmax(x, axis=-1) // 2
    one_hot_correct =tf.one_hot(index*2, num_skills*2)
    one_hot_incorrect = tf.one_hot(index*2+1, num_skills*2) 
    id_tensor = one_hot_correct + one_hot_incorrect
    # count features total/correct/incorrect tensor
    correct_count = tf.expand_dims(tf.reduce_sum(one_hot_correct * count_t, axis=-1), axis=-1)
    incorrect_count = tf.expand_dims(tf.reduce_sum(one_hot_incorrect * count_t, axis=-1), axis=-1)
    total_count = tf.expand_dims(tf.reduce_sum(id_tensor * count_t, axis=-1), axis=-1)
    all_count = tf.concat([correct_count, incorrect_count, total_count], axis=-1)
    count_feat = all_count

    embed_count = c_emb(count_feat)


    # *** log form 
    # count_t = tf.math.log1p(count_t)
    # embed_count = c_emb(count_t) # 2M to N-1

    # *** BEST learning constant learning curve 
    # embed_count = c_emb(count_t) # 2M to N-1
    # exp_count= tf.ones(shape=(tf.shape(embed_count))) - tf.math.exp(-embed_count)
    # coef_count = c_coef(exp_count)

    # *** forgetting curve
    seq_pre_delta = tf.math.log1p(seq_delta)
    rep_pre_delta = tf.math.log1p(rep_delta)
    # seq_pre_delta = seq_pre_delta * x 
    # rep_pre_delta = rep_pre_delta * x
    seq_embed_delta=seq_emb(seq_pre_delta)
    rep_embed_delta=rep_emb(rep_pre_delta)

    # *** preprocess for pure Pareto (log(1+raw t))
    # pre_delta = tf.math.log1p(delta)
    # embed_delta=delta_emb(pre_delta)
    # embed_delta=delta_emb(delta)
    # exp_delta= tf.math.exp(-embed_delta)

    # *** preprocess for coeff Pareto (log(1+mu*raw t))
    # coef_delta = delta_coeff(delta)
    # pre_delta = tf.math.log1p(coef_delta)
    # embed_delta=delta_emb(pre_delta)
    # embed_delta=delta_emb(delta)
    # exp_delta= tf.math.exp(-embed_delta)

    # *** BEST for log2_Parteto (log2(1+ t))preprocessed)
    # embed_delta=delta_emb(delta)
    # exp_delta= tf.math.exp(-embed_delta)

    # for Expo (mu*expo(-alpha*delta))
    # embed_delta=delta_emb(delta)
    # exp_delta= tf.math.exp(-embed_delta)
    # embed_exp_delta = delta_coeff(exp_delta)

    # *** concatenations
    x_c_t = c_concat([embed_x, seq_embed_delta, rep_embed_delta, embed_count]) # N+N
   
    # x_c_t = c_concat([embed_x, embed_count, exp_delta]) # N+N+1
    # x_c_t = c_concat([embed_x, exp_count, exp_delta]) # N+N+1
    # x_c_t = c_concat([embed_x, coef_count, exp_delta]) # N+N+1
    # x_c_t = c_concat([embed_x, embed_count, embed_exp_delta]) # N+N+1
    # x_c_t = c_concat([embed_count, exp_delta]) # N + 1 No x Cout + Delta_t
    # x_c_t = c_concat([embed_x, embed_count, embed_delta]) # N+N+1

    tempo_mask = count_mask.compute_mask(x)
    h = lstm(x_c_t, mask=tempo_mask)

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    outputs = reduce_sum(y_pred)

    # super().__init__(inputs=[x, delta, q], outputs=outputs, name="DKTAccumModel")
    super().__init__(inputs=[x, seq_delta, rep_delta, q], outputs=outputs, name="DKTAccum_no_count_Model")
