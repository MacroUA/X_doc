import pandas as pd
from X_model import *
from data.X_doc_tools import *
import matplotlib.pyplot as plt

data_path = 'C:/dataset/X_ray/images/'
df = pd.read_csv('./data/for_Meta.csv')
file = df.at[281,'Image Index']
file_path = data_path + file

model_name = 'Cardiomegaly8974_365365_1.h5'
model_path = './models/' + model_name
model.load_weights(model_path)

P, hm = heat_mapper2(model, img_path=file_path, inversion=model_name[-4], out_shape = out_shape)
#print(P, hm)


hm = hm.ravel()
side = int(len(hm)**.5)
hm *= hm
hm = hm.reshape(side, side)

panda_image = pd.DataFrame(hm)
panda_image.to_csv('./data/pandaim.csv')
plt.imshow(hm, cmap='coolwarm')

plt.show()