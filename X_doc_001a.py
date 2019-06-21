from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from data.X_doc_tools import *
import pandas as pd

#building model structure
if True:
    input_shape = (365, 365, 3)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')

model_name, diagnosis, invertion, save_mode = menu()

inp_path = './input/{}/'.format(diagnosis)
files_list = listdir(inp_path)

if diagnosis == 'Pneumonia':
    dfl = pd.read_csv('./data/pneumo_test.csv')
    dfl = dfl.loc[dfl['Image Index'].isin(files_list)]
else:
    dfl = pd.read_csv('./data/Data_Entry_2017.csv')
    dfl = dfl.loc[dfl['Image Index'].isin(files_list)]

dfb = pd.read_csv('data/BBox_List_2017.csv')
dfb = dfb.loc[dfb['Finding Label'] == diagnosis]

model.load_weights('./models/{}'.format(model_name))


for file in files_list:
    file_path = inp_path + file
    P, im, hm = heat_mapper(model=model, img_path=file_path, inversion=invertion)

    draw_report(P=P, img=im, hybrid=hm, dfl=dfl, file=file, model_name=model_name, dfb=dfb, diagnosis=diagnosis, save_mode=save_mode)
