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


def make_sequence_data(data_folder_path, processed_csv_dataname):
  """Make sequential data with temporal features (groupby user id) data and return it in Series

  Input:  path of Train dataset CSV file in the format below
                 <format>students id, skill id, correct, x(skill_id*2+1) 
  Output: sequential data in pandas Series
                <format> x, c_t(skill without the last attempt, seq_delta, repeated_delta, attempt_count),
                                    q, c_t+1, a
  """
  df = pd.read_csv(os.path.join(data_folder_path, processed_csv_dataname))

  # Get N, M, T
  num_students = df['user_id'].nunique()
  num_skills = df['skill_id'].nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  print(F"number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")

  seq = df.groupby('user_id').apply(
      lambda r: (
          r['x'].values[:-1], 

          r['skill_id'].values[:-1], # c_t
          r['seq_delta_t'].values[:-1],
          r['repeated_delta_t'].values[:-1],
          r['attempt_per_skill'].values[:-1],

          r['skill_id'].values[1:], # q

          r['seq_delta_t'].values[1:], # c_t+1
          r['repeated_delta_t'].values[1:],
          r['attempt_per_skill'].values[1:],

          r['correct'].values[1:], # a
      )
  )
  assert num_students, len(seq)

  return seq


def make_dkt_forget_2_seq(data_folder_path, processed_csv_dataname):

  df = pd.read_csv(os.path.join(data_folder_path, processed_csv_dataname))

    # Get N, M, T
  num_students = df['user_id'].nunique()
  num_skills = df['skill_id'].nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  print(F"number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")

  seq = df.groupby('user_id').apply(
      lambda r: (
          r['x'].values[:-1], # x
          r['seq_delta_t'].values[1:], # delta_t
          r['skill_id'].values[1:], # q_t
          r['correct'].values[1:], # a_t
      )
  )
  assert num_students, len(seq)
  return seq


def prepare_batched_tf_data(preprocessed_csv_seq, batch_size, num_skills, max_sequence_length):

  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: preprocessed_csv_seq,
      output_types=(tf.int32,
                    tf.int32, # skill
                    tf.float32, tf.float32, tf.float32, # c_t
                    tf.int32,
                    tf.float32, tf.float32, tf.float32, # c_t+1
                    tf.int32)
  )

  # One hot enconding
  # Encode binary sign of attmpts(from M to 2M)
  transformed_dataset  = dataset.map(
      lambda feat, skill_t, seq_d_t, rep_d_t, count_t, skill_t_1, seq_d_t_1, rep_d_t_1, count_t_1, label: (
          tf.one_hot(feat, depth=num_skills*2, dtype=tf.float32), # x  userT * 2M
          tf.one_hot(skill_t, depth=num_skills, axis=-1, dtype=tf.int32), # q
          tf.concat(
          [tf.math.multiply(tf.one_hot(skill_t, depth=num_skills, axis=-1, dtype=tf.float32), 
                           tf.repeat(tf.expand_dims(seq_d_t, axis=1), num_skills, axis=1)), # sequence_delta
          tf.math.multiply(tf.one_hot(skill_t, depth=num_skills, axis=-1, dtype=tf.float32), 
                           tf.repeat(tf.expand_dims(rep_d_t, axis=1), num_skills, axis=1)), # repeated_delta
          tf.math.multiply(tf.one_hot(skill_t, depth=num_skills, axis=-1, dtype=tf.float32), 
                           tf.repeat(tf.expand_dims(count_t, axis=1), num_skills, axis=1)) # counts of attempt
          ] ,1), # c_t
          tf.concat(
          [tf.math.multiply(tf.one_hot(skill_t_1, depth=num_skills, axis=-1, dtype=tf.float32), 
                           tf.repeat(tf.expand_dims(seq_d_t_1, axis=1), num_skills, axis=1)), # sequence_delta
          tf.math.multiply(tf.one_hot(skill_t_1, depth=num_skills, axis=-1, dtype=tf.float32), 
                           tf.repeat(tf.expand_dims(rep_d_t_1, axis=1), num_skills, axis=1)), # repeated_delta
          tf.math.multiply(tf.one_hot(skill_t_1, depth=num_skills, axis=-1, dtype=tf.float32), 
                           tf.repeat(tf.expand_dims(count_t_1, axis=1), num_skills, axis=1)) # counts of attempt
          ] ,1), # c_t+1
          tf.expand_dims(label, -1), # a 
      )
  )

  # Padding for LSTM
  # FIX: padding value should be Args and default -1
  padded_dataset = transformed_dataset.padded_batch(
          batch_size=batch_size,
          padding_values=(.0 ,0 ,.0, .0, -1),# -1),
          padded_shapes=([None, 2*num_skills], 
                         [None, num_skills], 
                         [None, 3*num_skills],
                         [None, 3*num_skills],
                         [None, 1],
                         )
      )
  

  # make mask for metrics
  padded_dataset  = padded_dataset.map(
    lambda x, delta_q, c_t, c_t_1, a: (
      x,
      delta_q,
      c_t,
      c_t_1,
      a,
      tf.cast(tf.math.logical_not(tf.math.equal(a, -1)),
              tf.float32) # mask for label
      )
  )


  # Dict format dataset to feed built-in function such as model.fit
  dict_dataset = padded_dataset.map(
          lambda x, delta_q, c_t, c_t_1, a, mask : (
              {"x" : x,
                "q" : delta_q,
               "c_t" : c_t,
               "c_t_1" : c_t_1,
               },
              { "outputs" : a},
              mask
          )
      )
  return dict_dataset  


def prepare_batched_tf_data_2(preprocessed_csv_seq, batch_size, num_skills, max_sequence_length):
   
  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: preprocessed_csv_seq,
      output_types=(
                    tf.int32, # x
                    tf.float32, # delta current
                    tf.int32, # q target skill
                    tf.int32) # a response
  )

  # One hot enconding
  # Encode binary sign of attmpts(from M to 2M)
  transformed_dataset  = dataset.map(
      lambda x, delta, q, label: (
          tf.one_hot(x, depth=num_skills*2), # x
        # tf.tile(tf.expand_dims(delta, -1), tf.constant([1,num_skills], tf.int32)), # expanded_delta
          tf.expand_dims(delta, -1),  # delta
          # tf.one_hot(q, depth=num_skills, on_value=1.0, off_value=0.0, axis=-1), # current skill positive(for counts)
          tf.one_hot(q, depth=num_skills, axis=-1), # q 
          tf.expand_dims(label, -1) # a 
      )
  )

  # Padding for LSTM
  # FIX: padding value should be Args and default -1
  padded_dataset = transformed_dataset.padded_batch(
          batch_size=batch_size,
          padding_values=(.0, .0, .0, -1),
          padded_shapes=(
                          [None, 2*num_skills],  # x          
                          # [max_sequence_length, num_skills], # expanded_delta
                          [None, 1], # delta
                          # [max_sequence_length, num_skills], # positive sign of skill
                          [None, num_skills], # q (skill of next step attempt)
                          [None, 1], # a
                          ),
  )

  # Dict format dataset to feed built-in function such as model.fit
  dict_dataset = padded_dataset.map(
          lambda x, delta, q, a : (
              {"x": x,
              "q" : q,
              "delta" : delta,
              #  "expanded_delta" : expanded_delta,
              #  "skill" : skill
                },
              { "outputs" : a}
          )
  )
  return dict_dataset  
