import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DKTforgetModel(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=100, hidden_units=100, dropout_rate=0.2):   
    x = tf.keras.Input(shape=(None, ), name='x')
    q = tf.keras.Input(shape=(None, num_skills), name='q')
    c_t = tf.keras.Input(shape=(None, num_skills*3), name='c_t')
    c_t_1 = tf.keras.Input(shape=(None, num_skills*3), name='c_t_1')

    x_emb = layers.Embedding(num_skills*2+1, embed_dim, 
                           mask_zero=True)

    c_emb = layers.Dense(embed_dim)
    c_concat = layers.Concatenate()

    mask = layers.Masking(mask_value=0.0)
    lstm =  layers.LSTM(hidden_units, return_sequences=True)
    out_dropout =  layers.TimeDistributed(layers.Dropout(dropout_rate))
    out_sigmoid =  layers.TimeDistributed(layers.Dense(num_skills,  activation='sigmoid'))
    dot =  layers.Multiply()
    # HACK: the shape of q does not fit to Timedistributed operation(may be correct?)
    # dot =  layers.TimeDistributed(layers.Multiply())

    reduce_sum =  layers.Dense(1, trainable=False, 
                                                      kernel_initializer=tf.keras.initializers.constant(value=1),
                                                      name="outputs")
    # reshape layer does not work as graph  # reshape_l = layers.Reshape((-1,6),dynamic=False)#, 

    # define graph
    n = x_emb(x)
    embed_c_t = c_emb(c_t)

    n = dot([n, embed_c_t])
    n = c_concat([n, c_t])

    masked_n = mask(n)
    h = lstm(masked_n)

    embed_c_t_1 = c_emb(c_t_1)
    h = dot([h, embed_c_t_1])
    h = c_concat([h, c_t_1])

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    # HACK: without using layer(tf.reduce) might be faster
    outputs = reduce_sum(y_pred)

    super().__init__(inputs=[x, q, c_t, c_t_1], outputs=outputs, name="DKTforgetModel")
