import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow import keras


def train_model(outfile_path, model, train_dataset, val_dataset, hparams, 
                num_students, num_skills, max_sequence_length, num_batches,
                num_hparam_search):

  # Start trainning
  print("-- start training --")
  print("hparams.hidden_units, hparams.dropout_rate, hparams.embed_dim, hparams.learning_rate,hparams.batch_size")
  print(hparams.hidden_units, hparams.dropout_rate, hparams.embed_dim, hparams.learning_rate,hparams.batch_size)
  print("num_students, num_skills, max_sequence_length, num_batches")
  print(num_students, num_skills, max_sequence_length, num_batches)

  # Create a TensorBoard callback
  model_name = model.__class__.__name__
  monitor_name = 'val_auc'

  early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor=monitor_name,
    patience=hparams.patience, 
    mode='max')
  reduce_lr_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_name, factor=0.5, patience=3, verbose=0, mode='max',
    min_delta=0.0001, cooldown=0, min_lr=0
  )

  # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") +"-"+  model_name
  logs = os.path.join(outfile_path, "keras_tensorboard", str(num_hparam_search+1)
)
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                   histogram_freq = 1,
                                                   update_freq='batch')
  # for debug  
  # history = model.fit(train_dataset.take(10),
  #                     epochs=hparams.num_epochs,
  #                     validation_data=val_dataset.take(10),
  #                     callbacks=[tboard_callback, early_stop_callback])

  history = model.fit(train_dataset.prefetch(5),
                      epochs=hparams.num_epochs,
                      validation_data=val_dataset.prefetch(5), 
                      callbacks=[tboard_callback,
                                 early_stop_callback,
                                 reduce_lr_plateau_callback]
  )
  print("-- finished training --")

  export_path = os.path.join(outfile_path, "keras_export")
  model.save(export_path)
  print('Model exported to: {}'.format(export_path))

  return max(history.history['val_auc']),  len(history.history['val_auc'])
