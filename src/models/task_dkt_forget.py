import argparse
import os
import json

import hypertune

import numpy as np
import pandas as pd
import tensorflow as tf

import models.deepkt_forget_tf2
from .train_dkt_forget_func import train_model
from data.tempo_tf_data_preprocessor import prepare_batched_tf_data, split_dataset, make_sequence_data, get_kfold_id_generator

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
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    # args_list = ['--job-dir', './', '--full_csv_dataname', './','--train_csv_dataname','./', '--cv_id_array_name','./']
    # args, _ = parser.parse_known_args(args_list)
    args, _ = parser.parse_known_args()
    return args


def do_one_time_cv_experiment(args):

  print("output directory: ", args.job_dir)
  print("Check GPUs", tf.config.list_physical_devices('GPU'))

  # Get N, M, T from a txt statstic file
  data_stats_dict = pd.read_json(os.path.join(args.data_folder_path, args.fulldata_stats_json_name),
                                                                    orient='columns')
  num_students = data_stats_dict['number of students'][0]
  num_skills = data_stats_dict['number of skills'][0]
  max_sequence_length = data_stats_dict['max attempt'][0]
  print(F"Full dataset info : number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")


  # prepare seq
  all_train_seq = make_sequence_data(args.data_folder_path, args.train_csv_dataname)

  # Get generator 
  num_fold=args.cv_num_folds
  scores = []
  steps = []
  kfold_index_gen = get_kfold_id_generator(all_train_seq, num_fold)

  for i in range(num_fold):
    train_index, val_index = next(kfold_index_gen)
    print(F"--Validation_set_number:{i+1}/{args.cv_num_folds}")
    print("TRAIN:", train_index, "TEST:", val_index)

    num_students = len(train_index)
    train_seq, val_seq = all_train_seq.iloc[train_index], all_train_seq.iloc[val_index]

    # prepare batch (padding, one_hot)
    train_tf_data = prepare_batched_tf_data(train_seq, args.batch_size, num_skills, max_sequence_length)
    val_tf_data = prepare_batched_tf_data(val_seq, args.batch_size, num_skills, max_sequence_length)
    num_batches = num_students // args.batch_size
    print(F"num_batches for training : {num_batches}")


    # build model
    model = models.deepkt_forget_tf2.DKTforgetModel(num_students, num_skills, max_sequence_length,
                              args.embed_dim, args.hidden_units, args.dropout_rate)
  
    # configure model
    # set Reduction.SUM for distributed traning
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
                metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.BinaryCrossentropy()]) # keep BCEntropyfor debug

    # KEEP: for debug 
    if i ==0:
      print("-- sample tf.data instance --")
      print(train_tf_data.take(1).element_spec)
    # sample = list(train_tf_data.take(1).as_numpy_iterator())
    # for i in range(3):
    #   print(sample[0][i])
    #   print(np.array(sample[0][i]).shape)
      print(model.summary()) 

    # start training
    max_score, global_step = train_model(args.job_dir, model, train_tf_data, val_tf_data, args,
                                                                                     num_students, num_skills, max_sequence_length,
                                                                                     num_batches, i)
    scores.append(max_score)
    steps.append(global_step)
    print("-- finished one fold --")

  #  Uses hypertune to report metrics for hyperparameter tuning.
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='val_auc',
      metric_value=sum(scores)/args.cv_num_folds,
      global_step=max(steps)
      )
  print("training result has been sent.")
 
  print("finished experiment")

if __name__ == '__main__':
    args = get_args()
    do_one_time_cv_experiment(args)
