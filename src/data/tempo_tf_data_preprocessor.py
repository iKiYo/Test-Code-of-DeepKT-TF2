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
# print(tf.test.is_gpu_available())
# preprocessed_csv_path ="/content/drive/My Drive/master thesis/Datasets/assistment_dataset/assist12_4cols_noNaNskill.csv" 
preprocessed_csv_path = "../../data/processed/assist12_4cols_noNaNskill.csv"
batch_size = 25

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
  # Todo: tuple converted to str, when read from csv
  # write out csv  file to storage
  # seq.to_csv(os.path.join(data_folder_path,"seq_"+processed_csv_dataname), index=False)

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
          r['skill_id'].values[:-1], # q_t
          r['correct'].values[1:], # a_t
      )
  )
  assert num_students, len(seq)
  return seq

def prepare_cv_id_array(data_folder_path, train_csv_dataname, num_fold):#, *hparams, *num_hparam_search):
  """ Make index array for X th fold cross validation splitting train/validation dataset
  Input: Train data in CSV
  Output: None, directly store the array in  .npy file to the same folder of the given train dataset
  """
  # Prepare seq series data
  all_train_seq = make_sequence_data(data_folder_path, train_csv_dataname)

  # Get generator 
  kfold_index_gen = get_kfold_id_generator(all_train_seq, num_fold)

  # Make ID array from generator and save it
  index_array= np.array([])
  for train_index, test_index in kfold_index_gen:
    # print("TRAIN:", train_index, "TEST:", test_index)
    index_array = np.append(index_array, [train_index ,test_index])
  index_array = index_array.reshape((num_fold,2))

  index_array_path = os.path.join(data_folder_path, 'cv_id_array_'+train_csv_dataname+'.npy')
  np.save(index_array_path, index_array)
  print(F"ID arrays cv_id_array.npy for CrossValidation saved to {index_array_path}")


def prepare_batched_tf_data(preprocessed_csv_seq, batch_size, num_skills, max_sequence_length):

  seq = preprocessed_csv_seq

  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: seq,
      # output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
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
          padded_shapes=([max_sequence_length, 2*num_skills], 
                         [max_sequence_length, num_skills], 
                         [max_sequence_length, 3*num_skills],
                         [max_sequence_length, 3*num_skills],
                         [max_sequence_length, 1],
                         ),
          drop_remainder=True
      )
  
  # Dict format dataset to feed built-in function such as model.fit
  dict_dataset = padded_dataset.map(
          lambda x, delta_q, c_t, c_t_1, a : (
              {"x" : x,
                "q" : delta_q,
               "c_t" : c_t,
               "c_t_1" : c_t_1,
               },
              { "outputs" : a}
          )
      )
  return dict_dataset  


def prepare_batched_tf_data_2(preprocessed_csv_seq, batch_size, num_skills, max_sequence_length):
 
  seq = preprocessed_csv_seq 
  
  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: seq,
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
                          [max_sequence_length, 2*num_skills],  # x          
                          # [max_sequence_length, num_skills], # expanded_delta
                          [max_sequence_length, 1], # delta
                          # [max_sequence_length, num_skills], # positive sign of skill
                          [max_sequence_length, num_skills], # q (skill of next step attempt)
                          [max_sequence_length, 1], # a
                          ),
          drop_remainder=True
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


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set
