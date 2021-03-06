"""
modified code from
https://github.com/lccasagrande/Deep-Knowledge-Tracing
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# print(tf.test.is_gpu_available())
# preprocessed_csv_path ="/content/drive/My Drive/master thesis/Datasets/assistment_dataset/assist12_4cols_noNaNskill.csv" 
preprocessed_csv_path = "../../data/processed/assist12_4cols_noNaNskill.csv"
batch_size = 25


def prepare_batched_tf_data(preprocessed_csv_path, batch_size=25):

  df = pd.read_csv(preprocessed_csv_path)

  # Get N, M, T
  # Todo: unique is not extensible
  num_students, num_skills, _, _ = df.nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  print(F"number of students:{num_students}\n number of skills:{num_skills}\n max of attempts sequence:{max_sequence_length}")

  seq = df.groupby('user_id').apply(
      lambda r: (
          r['x'].values[:-1], 
          r['skill_id'].values[1:],
          r['correct'].values[1:],
      )
  )

  assert num_students, len(seq)

  # Transform into tf.data format
  dataset = tf.data.Dataset.from_generator(
      generator=lambda: seq,
      output_types=(tf.int32, tf.int32, tf.int32)
  )

  # Shuffle before padding and making batches
  dataset.shuffle(buffer_size=num_students)

  # One hot enconding
  # Encode binary sign of attmpts(from M to 2M)
  transformed_dataset  = dataset.map(
      lambda feat, skill, label: (
          tf.one_hot(feat, depth=num_skills*2), # x
          tf.one_hot(skill, depth=num_skills, axis=-1), # q
          tf.expand_dims(label, -1) # a 
      )
  )

  # KEEP: for debug
  # padded_sample = list(transformed_dataset.take(3).as_numpy_iterator())
  # for i in range(3):
  #   print(padded_sample[0][i].T)
  #   print(np.array(padded_sample[0][i].T).shape)

  # Padding for LSTM
  # FIX: padding value should be Args and default -1
  padded_dataset = transformed_dataset.padded_batch(
          batch_size=batch_size,
          padding_values=(0.,0.,-1),
          padded_shapes=([max_sequence_length, 2*num_skills], [max_sequence_length, num_skills], [max_sequence_length, 1]),
          drop_remainder=True
      )

  # Dict format dataset to feed built-in function such as model.fit
  dict_dataset = padded_dataset.map(
          lambda x, delta_q, a : (
              {"x" : x,
                "q" : delta_q},
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

