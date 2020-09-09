import pandas as pd 

assist2012_preprocessed = pd.read_csv(
    "/content/drive/My Drive/master thesis/Datasets/assistment_dataset/2012-2013-data-with-predictions-4-final.csv",
    usecols=['user_id', 'skill_id', 'correct', 'end_time', 'attempt_count'])

print(len(assist2012_preprocessed))
assist2012_preprocessed.head(1)

df = assist2012_preprocessed

# drop NaN skill_ids 
df = df.dropna(subset = ['skill_id'])
# or fill NaN
#df = df.fillna(value={'skill_id': 999})
#df = df.astype({'skill_id': 'int32'})
print(len(df))
df.nunique()
df

df = df[df["correct"].isin([0, 1])]
df['correct'] = (df['correct'] >= 1).astype(int)

df

# delete user with only one attempt
counts = df['user_id'].value_counts()
df= df[~df['user_id'].isin(counts[counts < 2].index)]

# print(selected_df['user_id'].value_counts())
max_sequence_length=  df['user_id'].value_counts().max()
# print(max_sequence_length)
# print(selected_df['skill_id'].value_counts())
# sample entry
# df[df['user_id'] == 221324]
len(df.drop_duplicates()), len(df)

# sort by endtime(timestamp)
df =df.sort_values(by=['end_time'])
df

# enumerate and renumber the remain student and skill IDs
df['user_id'], _ = pd.factorize(df['user_id'], sort=False)
df['skill_id'], _ = pd.factorize(df['skill_id'], sort=False)

# get N, M, T
num_students, num_skills, _ = df.nunique()
assert max_sequence_length, df['user_id'].value_counts().max()

print(num_students, num_skills, max_sequence_length)

# Cross skill id with answer to form a synthetic feature
df['x'] = df['skill_id'] * 2 + df['correct']

df.to_csv("/content/drive/My Drive/master thesis/Datasets/assistment_dataset/assist12_4cols_noNaNskill.csv", index=False)
