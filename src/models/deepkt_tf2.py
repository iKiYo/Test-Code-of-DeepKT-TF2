import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# model parmaeters
hidden_units=200 
dropout_rate=0.2
embed_dim = 200
learning_rate = 0.005
# num_students, num_skills, max_sequence_length

# definition of input tensors and layers
x = tf.keras.Input(shape=(max_sequence_length, num_skills*2), name='x')
q = tf.keras.Input(shape=(max_sequence_length, num_skills), name='q')
emb =  layers.Dense(embed_dim, trainable=False, 
                                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=777),
                                                  input_shape=(None, max_sequence_length, num_skills*2))
mask = layers.Masking(mask_value=0, input_shape=(max_sequence_length, embed_dim))
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
layers.Masking(mask_value=0, input_shape=(None, max_sequence_length,1))
,name='outputs')


# define graph
n = emb(x)
masked_n = mask(n)
h = lstm(masked_n) 
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


# build model
model = keras.Model(inputs=[x, q], outputs=outputs)


# comile model
# set Reduction.SUM for distributed traning
model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
              optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                # metrics=[tf.keras.metrics.AUC()])
              metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug
