"""
modified code from
https://github.com/lccasagrande/Deep-Knowledge-Tracing
"""
import os 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold


def get_kfold_id_generator(array, num_fold):
  kf = KFold(n_splits = num_fold, shuffle = True, random_state = 2)
  return  kf.split(array)


def make_sequence_data(data_folder_path, processed_csv_dataname, window_size=None):
  """Make sequential(groupby user id) data and return it in Series

  Input:  path of Train dataset CSV file in the format below
                 <format>students id, skill id, correct, x(skill_id*2+1)
  Output: sequential data in pandas Series
 
  """
  # read and use preprocessed csv file
  df = pd.read_csv(os.path.join(data_folder_path, processed_csv_dataname))
  print(F"process :\n {processed_csv_dataname}  \nlength:  {len(df)}")


  # Get N, M, T
  num_students = df['user_id'].nunique()
  num_skills = df['skill_id'].nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  max_lag_time = np.unique(df['seq_delta_t'].values)[-2]
  print(F"number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")
  print(F"max lag time {max_lag_time}")

  if window_size is None:    
    raise  ValueError('No window size is specified')
  #   seq = df.groupby('user_id').apply(
  #       lambda r: (
  # # SAINT version input: Ex, a_shift, label: a
  #           r['skill_id'].values+1, # 1...T
  #           np.insert(r['correct'].values[:-1]+1, 0, [3]),# st_token + 1...T-1
  #           r['correct'].values,  # 1...T
  #           # r['seq_delta_t'].values,  # 1...T 
  #           # np.insert(r['seq_delta_t'].values[1:], 0, [0]) # st_token + 1...T-1        
  #           np.insert(r['seq_delta_t'].values[1:], 0, [max_lag_time+1]) # st_token + 1...T-1        
  #       )
  #   )
  else:
    df['attempt_per_user']  = df.groupby(['user_id']).cumcount()
    df['window_num'] = df['attempt_per_user']/window_size
    df = df.astype({'window_num': 'int32'}, copy=False)
    df['batch_num'] = df.groupby(['user_id','window_num']).ngroup()      
    num_batches = df['batch_num'].values.max()
    seq = df.groupby('batch_num').apply(
        lambda r: (
  # SAINT version input: Ex, a_shift, label: a
            r['skill_id'].values+1, # 1...T
            np.insert(r['correct'].values[:-1]+1, 0, [3]),# st_token + 1...T-1
            r['correct'].values,  # 1...T
            r['seq_delta_t'].values,  # 1...T 
            # np.insert(r['seq_delta_t'].values[1:], 0, [0]) # st_token + 1...T-1        
            # np.insert(r['seq_delta_t'].values[1:], 0, [max_lag_time+1]) # st_token + 1...T-1        
        )
    )

  assert num_students, len(seq)

  return seq, num_students, num_skills, max_sequence_length, num_batches


def prepare_batched_tf_data(preprocessed_csv_seq, batch_size, num_skills, max_sequence_length):

  seq = preprocessed_csv_seq

  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: seq,
      output_types=(tf.int32, tf.float32, tf.int32, tf.float32)
  )

  transformed_dataset  = dataset.map(
      lambda skill, shift_label, label, delta_t: (
          skill,
          shift_label,
          # delta_t,
          # tf.expand_dims(tf.math.log1p(delta_t), -1), # delta_t with log2 descretization
          tf.expand_dims(delta_t, -1), # delta_t 
          tf.expand_dims(label, -1) # a 
      )
  )

  # padded_dataset = dataset.padded_batch(
  padded_dataset = transformed_dataset.padded_batch(
          batch_size=batch_size,
          padding_values=(0, .0, .0,-1),  # padding is 0 but use (label) data - 1) 
          # padded_shapes=([None], [None], [None], [None, None]),
          padded_shapes=([None], [None], [None, 1], [None, 1]),
          # padded_shapes=([max_sequence_length], 
          #                [max_sequence_length],
          #                [max_sequence_length],
          #                [max_sequence_length, None]),
          drop_remainder=True
      )
  
  padded_dataset  = padded_dataset.map(
      lambda skill, shift_label, delta_t, label: (
          skill,
          shift_label,
          delta_t,
          label,
          tf.cast(tf.math.logical_not(tf.math.equal(label, -1)), tf.float32) # mask for label
      )
    )


  # # Dict format dataset to feed built-in function such as model.fit
  dict_dataset = padded_dataset.map(
          lambda skill, shift_label, delta_t, label, mask : (
              # SAINT version
              (skill, shift_label, delta_t),
              label,
              tf.squeeze(mask, -1)
          )
      )
  
  return dict_dataset

  # return padded_dataset
