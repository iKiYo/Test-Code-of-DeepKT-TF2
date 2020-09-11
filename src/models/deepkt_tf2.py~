import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# print(tf.test.is_gpu_available())

train_dataset, test_dataset, val_dataset = split_dataset(input_label_dataset, total_size=num_batches, test_fraction=0.1, val_fraction=0.2)

print(train_dataset.element_spec)
print(list(train_dataset.take(1).as_numpy_iterator()))
#print(list(test_dataset.take(1).as_numpy_iterator()))
#print(list(valid_dataset.take(1).as_numpy_iterator()))

# model parmaeters
hidden_units=200 
dropout_rate=0.2
embed_dim = 200
learning_rate = 0.005

num_students, num_skills, max_sequence_length

"""https://www.tensorflow.org/guide/keras/train_and_evaluate#passing_data_to_multi-input_multi-output_models"""

# definition of input tensor shape and layers
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

"""##### train model on assist12 dataset"""

model.predict(train_dataset.take(1))
model.summary()

"""###### measure the perfomace time"""

import time
class CustomCallback(keras.callbacks.Callback):
    # def on_train_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Starting training; got log keys: {}".format(keys))

    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))

    # def on_epoch_begin(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    # def on_epoch_end(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    # def on_test_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start testing; got log keys: {}".format(keys))

    # def on_test_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop testing; got log keys: {}".format(keys))

    # def on_predict_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start predicting; got log keys: {}".format(keys))

    # def on_predict_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        print(F" time {time.time()-self.batch_start_time :.3f} sec")

    def on_test_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_test_batch_end(self, batch, logs=None):
        print(F" time {time.time()-self.batch_start_time :.3f} sec")

    # def on_predict_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    # def on_predict_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


"""###### train"""

print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
print(num_students, num_skills, max_sequence_length, num_batches)

model.fit(train_dataset.prefetch(6),  epochs=1,  validation_data=val_dataset)#,  callbacks=[CustomCallback()])

"""###### evaluate with test dataset"""

results = model.evaluate(val_dataset.prefetch(5), callbacks=[CustomCallback()])
results

print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
print(num_students, num_skills, max_sequence_length, num_batches)

model.fit(train_dataset.prefetch(6),  epochs=1,  validation_data=val_dataset)#,  callbacks=[CustomCallback()])
# model.fit(input_label_dataset.take(200).prefetch(1), epochs=1, callbacks=[tboard_callback])

# model.evaluate(input_label_dataset.take(1))
