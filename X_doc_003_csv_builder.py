import pandas as pd


raw_csv = 'c:/dataset/X_ray/Data_Entry_2017.csv'
dnn_diag = ['No Finding', 'Cardiomegaly', 'Effusion', 'Hernia', 'Mass', 'Nodule', 'Pneumothorax']

df = pd.read_csv(raw_csv)[:]
df = df[df['View Position'] == 'PA'].reset_index(drop=True) #only PA  View Position
#df_NO = df[df['Finding Labels'] == 'No Finding'].reset_index(drop=True) #all PA NORMAL X_rays
#df_AB = df[df['Finding Labels'] != 'No Finding'].reset_index(drop=True) #all PA ABNORMAL X_rays

df2 = pd.DataFrame()

for col in ('Image Index',
            'Finding Labels',
            'Follow-up #',
            'Patient ID',
            'Patient Age',
            'Patient Gender',
            'View Position'):
    df2[col] = df[col]


for i in range(len(df2)):
    if i%100 == 0:
        print('converting {}%'.format(round(100/len(df2)*i,2)))

    found = 0
    for diagnosis in dnn_diag:
        if diagnosis in df2['Finding Labels'][i]:
            df2.at[i, diagnosis] = 1
            found = 1
        else:
            df2.at[i, diagnosis] = 0
    if found == 1:
        df2.drop(df2.index[i]) #delete rows wich not in dnn_diag

for diagnosis in dnn_diag:
    df2[diagnosis] = df2[diagnosis].astype(int)


print(df2)
df2.to_csv('./data/for_Meta.csv')

