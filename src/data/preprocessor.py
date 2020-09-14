import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data_file_name="2012-2013-data-with-predictions-4-final.csv"
data_folder_path = "/content/drive/My Drive/master thesis/Datasets/assistment_dataset/"
prepared_data_name="assist12_4cols_noNaNskill.csv"

data_file_path=data_folder_path + data_file_name
output_path = data_folder_path + prepared_data_name

  
def preprocess_csv(csv_file_path=data_file_path, out_path=output_path):
  # Load Assist2012 dataset, use "skill id"
  # Todo: read as int32 or 64
  
  df = pd.read_csv(data_file_path, usecols=['user_id', 'skill_id', 'correct', 'end_time'])

  # Drop NaN skill ids 
  df = df.dropna(subset = ['skill_id'])
  #df = df.fillna(value={'skill_id': 999}) # Or fill NaN skill
  df = df.astype({'skill_id': 'int32'})

  # Binarize correct values
  df = df[df["correct"].isin([0, 1])]
  df['correct'] = (df['correct'] >= 1).astype(int)

  # Delete user with only one attempt
  counts = df['user_id'].value_counts()
  df= df[~df['user_id'].isin(counts[counts < 2].index)]

  # Sort by endtime(timestamp)
  df =df.sort_values(by=['end_time']).drop(columns=['end_time'])

  # Enumerate and renumber the remain student and skill IDs
  df['user_id'], _ = pd.factorize(df['user_id'], sort=False)
  df['skill_id'], _ = pd.factorize(df['skill_id'], sort=False)

  # Show N, M, T
  num_students, num_skills = df[['user_id','skill_id']].nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  assert len(df.drop_duplicates()), len(df)
  print(F"number of students:{num_students}\n number of skills:{num_skills}\n max of attempts sequence:{max_sequence_length}")

  # Cross skill id with answer to form a synthetic feature
  df['x'] = df['skill_id'] * 2 + df['correct']

  df = df[['user_id', 'skill_id', 'correct']]
  df.to_csv(out_path, index=False)
  # df.to_csv("/content/drive/My Drive/master thesis/Datasets/assistment_dataset/assist12_4cols_noNaNskill.csv", index=False)
  print(F"preprocessed file is exported to \n{out_path}\n")


def make_train_test_csv(data_folder_path, csv_dataname, test_ratio):
  """Split a given csv dataset into train/test csv file

  Input:  path of full dataset CSV file in the  format below
                 <format>students id, skill id, correct, x(skill_id*2+1)
  Output: None, directory store train/test csv file in the same folder of input
  """
  # Read and use preprocessed csv file
  df = pd.read_csv(os.path.join(data_folder_path, csv_dataname))

  # Get N, M, T
  num_students = df['user_id'].nunique()
  num_skills = df['skill_id'].nunique()
  max_sequence_length=  df['user_id'].value_counts().max()
  print(F"number of students:{num_students}  number of skills:{num_skills}  max attempt :{max_sequence_length}")

  # Split whole data into train/test data
  train_index, test_index = train_test_split(np.arange(num_students), test_size=test_ratio, random_state=42)
  train_df = df[df['user_id'].isin(train_index)]
  test_df = df[df['user_id'].isin(test_index)]

  # Write them out to the same folder
  train_df.to_csv(os.path.join(data_folder_path,"train_"+csv_dataname), index=False)
  test_df.to_csv(os.path.join(data_folder_path,"test_"+csv_dataname), index=False)
