import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import deepkt_tf2
from data.tf_data_preprocessor import prepare_batched_tf_data, split_dataset
from data.preprocessor import preprocess_csv


def train_model(train_dataset, val_dataset,
                num_students, num_skills, max_sequence_length,
                batch_size, learning_rate,
                embed_dim, hidden_units, dropout_rate,
                num_batches, *num_hparam_search):
    
  # build model
  model = deepkt_tf2.DKTModel(num_students, num_skills, max_sequence_length,
                              embed_dim, hidden_units, dropout_rate)
  
  # configure model
  # set Reduction.SUM for distributed traning
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
                metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug

  # print(model.summary())

  # Start trainning
  print("start training")
  print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
  print(num_students, num_skills, max_sequence_length, num_batches)

  # Create a TensorBoard callback
  logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1, update_freq='batch')
  
  history = model.fit(train_dataset.prefetch(5),  epochs=25,  validation_data=val_dataset.prefetch(5), callbacks=[tboard_callback])

  
"""
# data path
csv_data_path = "../../data/processed/assist12_4cols_noNaNskill.csv"
# csv_data_path ="/content/drive/My Drive/master thesis/Datasets/assistment_dataset/assist12_4cols_noNaNskill.csv"


# data parameters
num_students=28118
num_skills=265
max_sequence_length=2089

# model parmaeters
hidden_units=200 
dropout_rate=0.2
embed_dim=200
learning_rate=0.005
batch_size=25

num_batches = num_students // batch_size

print(tf.config.list_physical_devices('GPU'))
# prepare tf data
batched_tf_data = prepare_batched_tf_data(preprocessed_csv_path=csv_data_path, batch_size=25)

# split the data
train_dataset, test_dataset, val_dataset = split_dataset(batched_tf_data, total_size=num_batches, test_fraction=0.1, val_fraction=0.2)

# build model
model = deepkt_tf2.DKTModel(num_students, num_skills, max_sequence_length, embed_dim, hidden_units, dropout_rate)

# configure model
# set Reduction.SUM for distributed traning
model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
              optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
              # metrics=[tf.keras.metrics.AUC()])
              metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug

print(model.summary())

# print(train_dataset.element_spec)
# print(list(train_dataset.take(1).as_numpy_iterator()))
#print(list(test_dataset.take(1).as_numpy_iterator()))
#print(list(valid_dataset.take(1).as_numpy_iterator()))

# model.predict(train_dataset.take(1))
# model.summary()

# Start trainning
print("start training")
print(hidden_units, dropout_rate, embed_dim, learning_rate,batch_size)
print(num_students, num_skills, max_sequence_length, num_batches)

history = model.fit(train_dataset.prefetch(5),  epochs=1,  validation_data=val_dataset)#,  callbacks=[CustomCallback()])
results = model.evaluate(val_dataset.prefetch(5))#, callbacks=[CustomCallback()])
print(results)
"""
