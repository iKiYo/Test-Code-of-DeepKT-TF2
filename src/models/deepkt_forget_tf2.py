import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DKTforgetModel(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=100, hidden_units=100, dropout_rate=0.2):   
    x = tf.keras.Input(shape=(max_sequence_length, num_skills*2), name='x')
    q = tf.keras.Input(shape=(max_sequence_length, num_skills), name='q')
    c_t = tf.keras.Input(shape=(max_sequence_length, num_skills*3), name='c_t')
    c_t_1 = tf.keras.Input(shape=(max_sequence_length, num_skills*3), name='c_t_1')

    emb =  layers.Dense(embed_dim, trainable=False, 
                                                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=777),
                                                      input_shape=(None, max_sequence_length, num_skills*2))
    
    c_emb = layers.Dense(embed_dim, trainable=True, 
                                                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=77),
                                                      input_shape=(None, max_sequence_length, num_skills*3))
    c_concat = layers.Concatenate()
    mask = layers.Masking(mask_value=0.0, input_shape=(max_sequence_length, embed_dim+num_skills*3))
    
    # mask = layers.Masking(mask_value=0, input_shape=(max_sequence_length, embed_dim))
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
    final_mask =   layers.TimeDistributed(
    layers.Masking(mask_value=0, input_shape=(None, max_sequence_length,1)),name='outputs')


    # define graph
    n = emb(x)

    embed_c_t = c_emb(c_t)
    n = dot([n, embed_c_t])
    n = c_concat([n, c_t])

    masked_n = mask(n)
    h = lstm(masked_n)

    embed_c_t_1 = c_emb(c_t_1)
    h = dot([h, embed_c_t_1])
    h = c_concat([h, c_t_1])
    h = mask(h) # repatch the mask

    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    # HACK: without using layer(tf.reduce) might be faster
    # y_pred = reduce_sum(y_pred, axis=2)
    y_pred = reduce_sum(y_pred)
    outputs = final_mask(y_pred)
    #  KEEP: another approach for final mask
    # patch initial mask by boolean_mask(tensor, mask)
    #tf.boolean_mask(y_pred, masked_n._keras_mask)
    #y_pred._keras_mask=masked_n._keras_mask

    super().__init__(inputs=[x, q, c_t, c_t_1], outputs=outputs, name="DKTforgetModel")
