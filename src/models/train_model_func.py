import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import deepkt_tf2
from data.tf_data_preprocessor import prepare_batched_tf_data, split_dataset
from data.preprocessor import preprocess_csv

import deepkt_tf2

def train_model(train_dataset, val_dataset, hparams, num_students, num_skills, max_sequence_length, num_batches, *num_hparam_search):
  # build model
  model = deepkt_tf2.DKTModel(num_students, num_skills, max_sequence_length,
                              hparams.embed_dim, hparams.hidden_units, hparams.dropout_rate)
  
  # configure model
  # set Reduction.SUM for distributed traning
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                optimizer=tf.optimizers.SGD(learning_rate=hparams.learning_rate),
                metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug

  print(model.summary())  

  # Start trainning
  print("-- start training --")
  print("hparams.hidden_units, hparams.dropout_rate, hparams.embed_dim, hparams.learning_rate,hparams.batch_size")
  print(hparams.hidden_units, hparams.dropout_rate, hparams.embed_dim, hparams.learning_rate,hparams.batch_size)
  print("num_students, num_skills, max_sequence_length, num_batches")
  print(num_students, num_skills, max_sequence_length, num_batches)

  # Create a TensorBoard callback
  model_name = model.__class__.__name__
  logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") +"-"+  model_name

  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1, update_freq='batch')
  # for debug  
  # history = model.fit(train_dataset.take(5).prefetch(5),  epochs=hparams.num-epochs,  validation_data=val_dataset.prefetch(5), callbacks=[tboard_callback])
  history = model.fit(train_dataset.prefetch(5),  epochs=hparams.num_epochs,  validation_data=val_dataset.prefetch(5), callbacks=[tboard_callback])
  print("-- finished training --")
