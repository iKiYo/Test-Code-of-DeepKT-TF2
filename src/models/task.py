from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from train_model_func import train_model
from data.tf_data_preprocessor import prepare_batched_tf_data, split_dataset, make_sequence_data

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
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--hidden_units',
        default=200,
        type=int,
        help='number of hidden units in LSTM cell, default=200')
    parser.add_argument(
        '--embed_dim',
        default=200,
        type=int,
        help='dimension of embedding in the first layer, default=200')
    parser.add_argument(
        '--dropout_rate',
        default=0.2,
        type=float,
        help='dropout_rate of outputs from LSTM cell, default=0.2')
    parser.add_argument(
        '--data_folder_path',
        default="../../data/processed",
        type=str,
        help='path of the dataset folder, default="../../data/processed"')
    parser.add_argument(
        '--full_csv_dataname',
        default=None,
        type=str,
        required=True,
        help='file name of Full dataset, default=None')
    parser.add_argument(
        '--train_csv_dataname',
        default=None,
        type=str,
        required=True,
        help='file name of Train dataset, default=None')
    parser.add_argument(
        '--cv_id_array_name',
        default=None,
        type=str,
        required=True,
        help='path to cross validation index array .npy file, default=None')   
    parser.add_argument(
        '--set_number',
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

  print("Check GPUs")
  print(tf.config.list_physical_devices('GPU'))

  # Get N, M, T from a full preprocessed csv file
  print("-- Get N, M, T from a full preprocessed csv file -- ")
  df = pd.read_csv(os.path.join(args.data_folder_path, args.full_csv_dataname))
  num_students = df['user_id'].nunique()
  num_skills = df['skill_id'].nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  print(F"Full dataset info : number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")


  # prepare seq
  all_train_seq = make_sequence_data(args.data_folder_path, args.train_csv_dataname)

  # Get CV id array
  cv_id_array = np.load(os.path.join(args.data_folder_path, args.cv_id_array_name), allow_pickle=True)
  print("--Check CV index --")
  print(cv_id_array[args.set_number-1][0])

  train_index  = cv_id_array[args.set_number-1][0]
  val_index = cv_id_array[args.set_number-1][1]
  num_students = len(train_index)
  train_seq, val_seq = all_train_seq.iloc[train_index], all_train_seq.iloc[val_index]

  # prepare batch (padding, one_hot)
  train_tf_data = prepare_batched_tf_data(train_seq, args.batch_size, num_skills, max_sequence_length)
  val_tf_data = prepare_batched_tf_data(val_seq, args.batch_size, num_skills, max_sequence_length)
  num_batches = num_students // args.batch_size
  print(F"num_batches for training : {num_batches}")

  # KEEP: for debug 
  print("-- sample tf.data instance --")
  print(train_tf_data.take(1).element_spec)
  # sample = list(train_tf_data.take(1).as_numpy_iterator())
  # for i in range(3):
  #   print(sample[0][i])
  #   print(np.array(sample[0][i]).shape)

  # start training
  train_model(train_tf_data, val_tf_data, args, num_students, num_skills, max_sequence_length, num_batches, 1)
  print("finished experiment")

if __name__ == '__main__':
    args = get_args()
    do_one_time_cv_experiment(args)
