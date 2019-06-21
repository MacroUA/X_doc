import pandas as pd
import shutil

# config
diagnosis = 'Nodule'
abnormal_csv = './data/BBox_List_2017.csv'
#abnormal_csv = './data/Data_Entry_2017.csv'
normal_csv = './data/Data_Entry_2017.csv'
max_nb_of_files = 20000 #for each normal and abnormal total =* 2
shuffle_df = True

dir_from = 'C:/X_ray/data/images/' #directory of dataset
dir_to = './input/{}/'.format(diagnosis)

df = pd.read_csv(abnormal_csv)
df = df.loc[df['Finding Label'] == diagnosis]
if shuffle_df:
    df = df.sample(frac=1).reset_index(drop=True)

df_n = pd.read_csv(normal_csv)
df_n = df_n.loc[df_n['Finding Labels'] == 'No Finding']
if shuffle_df:
    df_n = df_n.sample(frac=1).reset_index(drop=True)

dflen = len(df)
if dflen > max_nb_of_files:
    dflen = max_nb_of_files


for i in range(dflen):
    print('copy files ', int(100/dflen*i), '%')
    filename = df.iloc[i]['Image Index']
    filename_nf = df_n.iloc[i]['Image Index']

    shutil.copy(dir_from + filename, dir_to)
    shutil.copy(dir_from + filename_nf, dir_to)

