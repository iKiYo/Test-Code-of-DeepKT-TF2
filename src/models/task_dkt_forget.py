import argparse
import os
import json
import time
import hypertune

import numpy as np
import pandas as pd
import tensorflow as tf

import models.deepkt_forget_tf2
from .train_model import train_model
from data.tempo_tf_data_preprocessor import prepare_batched_tf_data, make_sequence_data, get_kfold_id_generator

def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=30,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.001,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--hidden_units',
        default=100,
        type=int,
        help='number of hidden units in LSTM cell, default=100')
    parser.add_argument(
        '--embed_dim',
        default=100,
        type=int,
        help='dimension of embedding in the first layer, default=100')
    parser.add_argument(
        '--dropout_rate',
        default=0.2,
        type=float,
        help='dropout_rate of outputs from LSTM cell, default=0.2')
    parser.add_argument(
        '--data_folder_path',
        default="../data/processed",
        type=str,
        help='path of the dataset folder, default="../../data/processed"')
    parser.add_argument(
        '--full_csv_dataname',
        default="assist12_8cols_log2_noNaNskill.csv",
        type=str,
        help='file name of Full dataset, default=None')
    parser.add_argument(
        '--fulldata_stats_json_name',
        default="assist12_8cols_log2_noNaNskill.json",
        type=str,
        required=True,
        help='json file of Full data statistic, default=None')
    parser.add_argument(
        '--train_csv_dataname',
        default="train_assist12_8cols_log2_noNaNskill.csv",
        type=str,
        required=True,
        help='file name of Train dataset, default=None')
    parser.add_argument(
        '--test_csv_dataname',
        default=None,
        type=str,
        help='file name of Test dataset, default=None')
    parser.add_argument(
        '--cv_num_folds',
        default=5,
        type=int,
        help='number of cross-validation folds, default=5')
    parser.add_argument(
        '--cv_set_number',
        default=1,
        type=int,
        help='cross validation dataset number, default=1')
    parser.add_argument(
        '--num_trial',
        default=1,
        type=int,
        help='number of trail for one-time experiment, default=1')
    parser.add_argument(
        '--LR_test',
        default=0.0,
        type=float,
        help='execute learning rate test, default=0')
    parser.add_argument(
        '--patience',
        default=7,
        type=int,
        help='early stopping epochs, default=7')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    # args_list = ['--job-dir', './', '--full_csv_dataname', './','--train_csv_dataname','./', '--cv_id_array_name','./']
    # args, _ = parser.parse_known_args(args_list)
    args, _ = parser.parse_known_args()
    return args


def get_full_data_stats(args):

  # Get N, M, T from a txt statstic file
  data_stats_dict = pd.read_json(os.path.join(args.data_folder_path, args.fulldata_stats_json_name),
                                                                    orient='columns')
  num_students = data_stats_dict['number of students'][0]
  num_skills = data_stats_dict['number of skills'][0]
  max_sequence_length = data_stats_dict['max attempt'][0]
  print(F"Full dataset info : number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")
  return num_students, num_skills, max_sequence_length


def do_one_time_cv_experiment(args, num_students, num_skills, max_sequence_length):
  print(args)
  # prepare seq
  all_train_seq = make_sequence_data(args.data_folder_path, args.train_csv_dataname)

  # Get generator 
  num_fold=args.cv_num_folds
  kfold_index_gen = get_kfold_id_generator(all_train_seq, num_fold)


  loss=tf.keras.losses.BinaryCrossentropy(
      reduction=tf.keras.losses.Reduction.SUM)
  auc, bce = tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy() 


   # start trainings
  scores = []
  steps = []
  elapsed_time = []

  for i in range(num_fold):
    start = time.perf_counter()
    train_index, val_index = next(kfold_index_gen)
    print(F"--Validation_set_number:{i+1}/{args.cv_num_folds}")
    # print("TRAIN:", train_index, "TEST:", val_index)

    num_students = len(train_index)
    train_seq, val_seq = all_train_seq.iloc[train_index], all_train_seq.iloc[val_index]

    # prepare batch (padding, one_hot)
    train_tf_data = prepare_batched_tf_data(train_seq,
                                            args.batch_size,
                                            num_skills,
                                            max_sequence_length
    )
    val_tf_data = prepare_batched_tf_data(val_seq,
                                          args.batch_size,
                                          num_skills,
                                          max_sequence_length
    )
    num_batches = num_students // args.batch_size
    print(F"num_batches for training : {num_batches}")


    # build model
    model = models.deepkt_forget_tf2.DKTforgetModel(num_students, num_skills,
                                       max_sequence_length,
                                       args.embed_dim, args.hidden_units,
                                       args.dropout_rate)
    
    # LR test setting
    learning_rate = args.learning_rate
    if args.LR_test != 0.0:
      print(F"lr {args.LR_test}")
      args.num_epochs=1

      if args.LR_test == 1e-6:
        init_lr = 1e-6
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            init_lr,
            decay_steps=1,
            decay_rate=10,
            staircase=True)
        train_tf_data = train_tf_data.take(6)
        val_tf_data = val_tf_data.take(6)

      if args.LR_test > 1e-6:
        init_lr = args.LR_test
        range_list = np.arange(init_lr, init_lr*10+init_lr, init_lr).tolist()
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=np.arange(1, len(range_list)).tolist(), 
            values=range_list,
        )
        train_tf_data = train_tf_data.take(10)
        val_tf_data = val_tf_data.take(10)

    # configure model
    # set Reduction.SUM for distributed traning
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate)
    auc.reset_states()
    bce.reset_states()
    model.compile(optimizer, loss, weighted_metrics=[auc, bce])
    
    # KEEP: for debug 
    if i ==0:
      print("-- sample tf.data instance --")
      print(train_tf_data.take(1).element_spec)
    # sample = list(train_tf_data.take(1).as_numpy_iterator())
    # for i in range(3):
    #   print(sample[0][i])
    #   print(np.array(sample[0][i]).shape)
      print(model.summary()) 

    max_score, global_step = train_model(args.job_dir, model,
                                         train_tf_data, val_tf_data, args,
                                         num_students, num_skills,
                                         max_sequence_length,
                                         num_batches, i)
    scores.append(max_score)
    steps.append(global_step)
    elapsed_time.append(time.perf_counter() - start)
    print(F"-- finished {i+1}/{num_fold} --")
    print("-- finished one fold --")

    
  df = pd.DataFrame({'Trial ID': range(1,num_fold+1),
                     'val_auc':scores, 'Training step':steps,
                     ' Elapsed time ': elapsed_time,
                     ' learning-rate': [args.learning_rate]*num_fold})
  df.round(5).to_csv(os.path.join(args.job_dir, "result_table.csv"))

  #  Uses hypertune to report metrics for hyperparameter tuning.
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='val_auc',
      metric_value=sum(scores)/args.cv_num_folds,
      global_step=max(steps)
      )
  print("training result has been sent.")
  
  print("finished experiment")


def do_normal_experiment(args, num_students, num_skills, max_sequence_length):

  train_seq = make_sequence_data(args.data_folder_path, args.train_csv_dataname)
  val_seq = make_sequence_data(args.data_folder_path, args.test_csv_dataname)

  # prepare batch (padding, one_hot)
  train_tf_data = prepare_batched_tf_data(train_seq,
                                          args.batch_size,
                                          num_skills,
                                        max_sequence_length
  )
  val_tf_data = prepare_batched_tf_data(val_seq,
                                        args.batch_size,
                                        num_skills,
                                        max_sequence_length
  )
  num_batches = num_students // args.batch_size
  print(F"num_batches for training : {num_batches}")

  # build model
  model = models.deepkt_forget_tf2.DKTforgetModel(num_students, num_skills, max_sequence_length,
                            args.embed_dim, args.hidden_units, args.dropout_rate)

  loss=tf.keras.losses.BinaryCrossentropy(
      reduction=tf.keras.losses.Reduction.SUM)
  optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
  auc, bce = tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy() 

   # start trainings
  scores = []
  steps = []
  elapsed_time = []
  for i in range(args.num_trial):
    start = time.perf_counter()

    # set Reduction.SUM for distributed traning
    auc.reset_states()
    bce.reset_states()
    model.compile(loss, optimizer, weighted_metrics=[auc, bce]) # keep BCEntropyfor debug

    # KEEP: for debug 
    print("-- sample tf.data instance --")
    print(train_tf_data.take(1).element_spec)
    # sample = list(train_tf_data.take(1).as_numpy_iterator())
    # for i in range(3):
    #   print(sample[0][i])
    #   print(np.array(sample[0][i]).shape)
    print(model.summary()) 

    # start training
    max_score, global_step = train_model(args.job_dir, model,
                                         train_tf_data, val_tf_data, args,
                                         num_students, num_skills,
                                         max_sequence_length,
                                         num_batches, i)
    scores.append(max_score)
    steps.append(global_step)
    elapsed_time.append(time.perf_counter() - start)
    print(F"-- finished {i+1}/{args.num_trial} --")

    
  df = pd.DataFrame({'Trial ID': range(1,args.num_trial+1),
                     'val_auc':scores, 'Training step':steps,
                     ' Elapsed time ': elapsed_time,
                     ' learning-rate': [args.learning_rate]*3})
  df.round(5).to_csv(os.path.join(args.job_dir, "result_table.csv"))

  #  Uses hypertune to report metrics for hyperparameter tuning.
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='val_auc',
      metric_value=sum(scores)/args.num_trial,
      global_step=steps[scores.index(max(scores))]
      )
  print("training result has been sent.") 
  print("finished experiment")


if __name__ == '__main__':
    args = get_args()
    num_students, num_skills, max_sequence_length = get_full_data_stats(args)
    print("output directory: ", args.job_dir)
    print("Check GPUs", tf.config.list_physical_devices('GPU'))
    if args.test_csv_dataname is None:
      do_one_time_cv_experiment(args, num_students, num_skills,
                                max_sequence_length)
    else:
      do_normal_experiment(args, num_students, num_skills, max_sequence_length)
