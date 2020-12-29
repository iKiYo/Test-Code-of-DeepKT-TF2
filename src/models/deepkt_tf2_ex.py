import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DKTModel(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length, embed_dim=200,
                            hidden_units=100, dropout_rate=0.2):   

    # x = tf.keras.Input(shape=(None,), name='x')
    x = tf.keras.Input(shape=(None,num_skills*2+1), name='x')
    q = tf.keras.Input(shape=(None, num_skills), name='q')

    # emb = layers.Embedding(num_skills*2+1, embed_dim, 
    #                        embeddings_initializer=\
    #                        tf.keras.initializers.RandomNormal(seed=777),
    #                        input_length=max_sequence_length,
    #                        mask_zero=True)

    emb = layers.Dense(102, activation='softmax')
    # emb = layers.Dense(num_skills*2+1, activation='softmax')
    # dense_soft = layers.Dense(embed_dim, activation='softmax',
    #                                                        trainable=False,
    #                                                       kernel_initializer=\
    #                             tf.keras.initializers.constant(value=1)
    #   )                       

    # soft = layers.Softmax(input_shape=(None, embed_dim))
    mask = layers.Masking(mask_value=0.0)
    # emb = layers.Dense(embed_dim, activation='softmax')

    # sk_emb = layers.Dense(102)

    lstm = layers.LSTM(hidden_units, return_sequences=True)
    out_dropout = layers.TimeDistributed(layers.Dropout(dropout_rate))
    out_sigmoid = layers.TimeDistributed(layers.Dense(num_skills,
                                                      activation='sigmoid'))
    dot = layers.Multiply()
    # HACK: the shape of q does not fit to Timedistributed operation
    # dot =  layers.TimeDistributed(layers.Multiply())
    reduce_sum = layers.Dense(1, trainable=False, 
                              kernel_initializer=\
                              tf.keras.initializers.constant(value=1),
                              input_shape=\
                              (None, max_sequence_length, num_skills),
                              name="outputs")
    # reshape layer does not work as graph
    # reshape_l = layers.Reshape((-1,6),dynamic=False)#, 
 
    # define graph

    # without keras Emblyaer
    mask_x = mask(x)
    n = emb(mask_x)
    # emb_x = emb(mask_x)
    # n = sk_emb(emb_x)

    # n_x = emb(x)
    # n = soft(n_x)
    # n = dense_soft(n_x)
    # tf.shape(n)
    # n = soft(n_x, mask=emb.compute_mask(x))
    # n = mask(n)
    # soft_n_x = soft(n_x)
    # n = sk_emb(soft_n_x)

    # n = emb(x)
    h = lstm(n) 
    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    # HACK: without using layer(tf.reduce) might be faster
    outputs = reduce_sum(y_pred)
  
    super().__init__(inputs=[x, q], outputs=outputs, name="DKTModel")
