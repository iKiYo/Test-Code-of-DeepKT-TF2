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
  print(F"number of students:{num_students}  number of skills:{num_skills}"
        F" max attempt :{max_sequence_length}")

  seq = df.groupby('user_id').apply(
      lambda r: (
          r['x'].values[:-1]+1, 
          r['skill_id'].values[1:],
          r['correct'].values[1:],
      )
  )

  assert num_students, len(seq)
  
  return seq, num_students, num_skills, max_sequence_length
  

def prepare_cv_id_array(data_folder_path, train_csv_dataname, num_fold):
  #, *hparams, *num_hparam_search):
  """ Make index array for X th fold cross validation splitting train/validation
 dataset
  Input: Train data in CSV
  Output: None, directly store the array in  .npy file to the same folder of 
  the given train dataset
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

  index_array_path = os.path.join(data_folder_path,
                                  'cv_id_array_'+train_csv_dataname+'.npy')
  np.save(index_array_path, index_array)
  print(F"ID arrays cv_id_array.npy for CrossValidation saved to {index_array_path}")


def prepare_batched_tf_data(preprocessed_csv_seq, batch_size, num_skills,
                            max_sequence_length):

  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: preprocessed_csv_seq,
      output_types=(tf.int32, tf.int32, tf.int32)
  )

  # todo: no need? because it is shuffled  when split
  # Shuffle before padding and making batches
  # dataset.shuffle(buffer_size=num_students)

  # One hot enconding
  # Encode binary sign of attmpts(from M to 2M)
  transformed_dataset  = dataset.map(
      lambda feat, skill, label: (
          feat, # x 
          tf.one_hot(skill, depth=num_skills, axis=-1), # q
          tf.expand_dims(label, -1) # a 
      )
  )

  # KEEP: for debug
  # padded_sample = list(transformed_dataset.take(3).as_numpy_iterator())
  # for i in range(3):
  #   print(padded_sample[0][i].T)

  # Padding for LSTM
  padded_dataset = transformed_dataset.padded_batch(
    batch_size=batch_size,
    padding_values=(0, 0., -1),
    padded_shapes=([None], 
                   [None, num_skills],
                   [None, 1]),
  )

  # make mask for metrics
  padded_dataset  = padded_dataset.map(
    lambda x, skill, label: (
      x,
      skill,
      label,
      tf.cast(tf.math.logical_not(tf.math.equal(label, -1)),
              tf.float32) # mask for label
      )
  )

  # Dict format dataset to feed built-in function such as model.fit
  dict_dataset = padded_dataset.map(
    lambda x, q, a, mask : (
      {"x" : x,
       "q" : q},
      {"outputs" : a},
      mask
    )
  )
  
  return dict_dataset
