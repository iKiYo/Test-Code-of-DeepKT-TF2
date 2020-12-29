import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DKTModel(tf.keras.Model):


  def __init__(self, num_students, num_skills, max_sequence_length,
               embed_dim=200, hidden_units=100, dropout_rate=0.2):   

    x = tf.keras.Input(shape=(None,), name='x')
    q = tf.keras.Input(shape=(None, num_skills), name='q')

    emb = layers.Embedding(num_skills*2+1, embed_dim, 
                           embeddings_initializer=\
                           tf.keras.initializers.RandomNormal(seed=777),
                           input_length=max_sequence_length,
                           mask_zero=True)

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
    n = emb(x)
    h = lstm(n) 
    o = out_dropout(h)
    y_pred = out_sigmoid(o)
    y_pred = dot([y_pred, q])
    # HACK: without using layer(tf.reduce) might be faster
    outputs = reduce_sum(y_pred)
  
    super().__init__(inputs=[x, q], outputs=outputs, name="DKTModel")
