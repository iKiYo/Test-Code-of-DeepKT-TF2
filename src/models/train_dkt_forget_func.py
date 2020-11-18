import os
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

import hypertune

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import models.deepkt_forget_tf2
from data.tempo_tf_data_preprocessor import prepare_batched_tf_data, split_dataset
from data.preprocessor import preprocess_csv


def train_model(outfile_path, model, train_dataset, val_dataset, hparams, 
                                   num_students, num_skills, max_sequence_length, num_batches, num_hparam_search):
  # build model
  # model = models.deepkt_forget_tf2.DKTforgetModel(num_students, num_skills, max_sequence_length,
                              # hparams.embed_dim, hparams.hidden_units, hparams.dropout_rate)
  
  # configure model
  # set Reduction.SUM for distributed traning
  # model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                # optimizer=tf.optimizers.Adam(learning_rate=hparams.learning_rate),
                # metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug

  # print(model.summary())  

  # Start trainning
  print("-- start training --")
  print("hparams.hidden_units, hparams.dropout_rate, hparams.embed_dim, hparams.learning_rate,hparams.batch_size")
  print(hparams.hidden_units, hparams.dropout_rate, hparams.embed_dim, hparams.learning_rate,hparams.batch_size)
  print("num_students, num_skills, max_sequence_length, num_batches")
  print(num_students, num_skills, max_sequence_length, num_batches)

  # Create a TensorBoard callback
  model_name = model.__class__.__name__
  if num_hparam_search == 0:
    monitor_name = 'val_auc'
  else:
    monitor_name = 'val_auc_'+str(num_hparam_search)

  early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor_name, min_delta=0.01, patience=3, 
                                                                                                                    mode='max')

  # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S") +"-"+  model_name
  logs = os.path.join(outfile_path, "keras_tensorboard")
  tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1)#, update_freq='batch')
  # for debug  
  # history = model.fit(train_dataset.take(5),  epochs=hparams.num_epochs,  validation_data=val_dataset.take(3), callbacks=[tboard_callback, early_stop_callback])
  history = model.fit(train_dataset.prefetch(5),  epochs=hparams.num_epochs,
                                         validation_data=val_dataset, steps_per_epoch=num_batches//1000,
                                         validation_steps =num_batches//1000,
                                         callbacks=[tboard_callback,early_stop_callback])
  print("-- finished training --")

   # Uses hypertune to report metrics for hyperparameter tuning.
  # hpt = hypertune.HyperTune()
  # hpt.report_hyperparameter_tuning_metric(
  #     hyperparameter_metric_tag='val_auc',
  #     metric_value=max(history.history['val_auc']),
  #     global_step=len(history.history['val_auc']))
  # print("training reuslt has been sent.")
 
  export_path = os.path.join(outfile_path, "keras_export")
  model.save(export_path)
  print('Model exported to: {}'.format(export_path))

  if num_hparam_search == 0:
    return max(history.history['val_auc']),  len(history.history['val_auc'])
  else:
    return max(history.history['val_auc_'+str(num_hparam_search)]), len(history.history['val_auc_'+str(num_hparam_search)])
