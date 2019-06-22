import keras.backend as K
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os import listdir

out_shape = (640, 640)

#just a Menu
def menu():
    print('\n'*4)
    print('╔═════════════════════════════════════╗')
    print('║       AI Radiologist 0.01a          ║')
    print('╟─────────────────────────────────────╢')
    print('║ by Andrii Oleksandrovych Sydorenko  ║')
    print('║      & Ricardo Teresa Ribeiro       ║')
    print('╚═════════════════════════════════════╝')
    print('\n' * 1)
    print('Available Neural Networks:')
    print('\n' * 1)
    models = listdir('./models/')
    for i, m in enumerate(models):
        print(i, m)
    print('\n' * 1)
    done = False
    while not done:
        try:
            model_num = int(input('choose a model:'))
            if model_num in list(range(len(models))):
                done = True
        except Exception:
            print('\n')
            print('please repeat your choice')

    print('\n' * 2)
    save_mode = None
    while save_mode != 0 and save_mode != 1:
        try:
            save_mode = int(input('select output( 0 display / 1 files):'))
        except Exception:
            print('\n' * 1)
            print('please repeat your choice')
    print('\n' * 4)
    model_name = models[model_num]
    diagnosis = models[model_num].split('_')[0][:-4]
    invertion = int(models[model_num].split('_')[-1][:-3])

    return model_name, diagnosis, invertion, save_mode

#generate X
def X_from_img(path, size = (365, 365)):
    img_path = path
    img = image.load_img(img_path, target_size=size)

    img = image.img_to_array(img)/255

    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    #img = preprocess_input(img)
    return img

#get Label from csv
def label_from_index(dataframe, filename):
    dataframe = dataframe.loc[dataframe['Image Index'] == filename]
    return dataframe.iloc[0]['Finding Labels']

#get Bbox from CSV
def bbox_from_index(dataframe, filename):
    dataframe = dataframe.loc[dataframe['Image Index'] == filename]
    x = dataframe.iloc[0]['Bbox [x']
    y = dataframe.iloc[0]['y']
    w = dataframe.iloc[0]['w']
    h = dataframe.iloc[0]['h]']
    # xy = (df['Bbox [x'][i] / 1024 * out_shape[0], df['y'][i] / 1024 * out_shape[1])
    # w = df['w'][i] / 1024 * out_shape[0]
    # h = df['h]'][i] / 1024 * out_shape[1]
    return x, y, w, h

#make heatmap from file
def heat_mapper(model, img_path, inversion=0, out_shape = out_shape):
    X = X_from_img(img_path)
    P = model.predict(X)

    class_idx = 0
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("conv2d_13")
    grads = K.gradients(class_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([X])

    for o in range(len(pooled_grads_value)):  # 512
        #   print(pooled_grads_value[i])
        conv_layer_output_value[:, :, o] *= pooled_grads_value[o]



    img = cv2.imread(img_path)
    img = cv2.resize(img, out_shape)

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    if inversion:
        heatmap = heatmap*-1
        P = 1-P
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, out_shape)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    hybrid = cv2.addWeighted(img, 1, heatmap, int((P > 0.0)), 0)

    return P, img, hybrid

#make heatmap from file
def heat_mapper2(model, img_path, inversion=0, out_shape = out_shape):
    X = X_from_img(img_path)
    P = model.predict(X)

    class_idx = 0
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("conv2d_13")
    grads = K.gradients(class_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([X])

    for o in range(len(pooled_grads_value)):  # 512
        #   print(pooled_grads_value[i])
        conv_layer_output_value[:, :, o] *= pooled_grads_value[o]



    img = cv2.imread(img_path)
    img = cv2.resize(img, out_shape)

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    if inversion:
        heatmap = heatmap*-1
        P = 1-P

    #heatmap2 = np.maximum(heatmap, 0) # delete heatmap is <0

    heatmap -= np.min(heatmap) #heatmap from 0
    heatmap /= np.max(heatmap) #heatmap to 1
    print(np.min(heatmap), np.max(heatmap))
    print(P)

    #heatmap = cv2.resize(heatmap, out_shape)
    #heatmap = np.uint8(255 * heatmap)
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    #hybrid = cv2.addWeighted(img, 1, heatmap, int((P > 0.0)), 0)



    return P, heatmap


#visualizator
def draw_report(P, img, hybrid, dfl, file, model_name, dfb, diagnosis, save_mode=0):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    true_diag = label_from_index(dfl, file)
    ax[0].set_title('True diagnosis:\n'+true_diag)
    if P > 0.5:
        ax[1].set_title('Diagnosed with AI:\n{} {}%'.format(diagnosis, round(float(P[0]) * 100, 2)))
    else:
        ax[1].set_title('Diagnosed with AI:\n{}'.format('No finding'))



    ax[0].imshow(img, cmap='binary')
    ax[1].imshow(hybrid, cmap='binary')

    fig.suptitle('file: {}\nmodel: {}'.format(file, model_name), fontsize=16)

    try:
        x, y, w, h = bbox_from_index(dfb, file)
        xy = (x / 1024 * out_shape[0], y / 1024 * out_shape[1])
        w = w / 1024 * out_shape[0]
        h = h / 1024 * out_shape[1]
        ax[0].add_patch(patches.Rectangle(xy, w, h, linewidth=1, edgecolor='r', facecolor='none'))
    except Exception:
        print('no BBOX found')

    print(file)

    if save_mode:
        fig.savefig('./output/{}/{}.png'.format(diagnosis, str(int(P * 10000)) + '_' + file[:-4] ), dpi=100)
    else:
        plt.show()

#visualizator
def draw_report2(P, img, hybrid, file, model_name, diagnosis, save_mode=0):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    true_diag = 'Unknown'
    ax[0].set_title('True diagnosis:\n'+true_diag)
    if P > 0.0:
        ax[1].set_title('Diagnosed with AI:\n{} {}%'.format(diagnosis, round(float(P[0]) * 100, 2)))
    else:
        ax[1].set_title('Diagnosed with AI:\n{}'.format('No finding'))



    ax[0].imshow(img, cmap='binary')
    ax[1].imshow(hybrid, cmap='binary')

    fig.suptitle('file: {}\nmodel: {}'.format(file, model_name), fontsize=16)

    try:
        x, y, w, h = bbox_from_index(dfb, file)
        xy = (x / 1024 * out_shape[0], y / 1024 * out_shape[1])
        w = w / 1024 * out_shape[0]
        h = h / 1024 * out_shape[1]
        ax[0].add_patch(patches.Rectangle(xy, w, h, linewidth=1, edgecolor='r', facecolor='none'))
    except Exception:
        print('no BBOX found')

    print(file)

    if save_mode:
        fig.savefig('./output/lis/{}.png'.format(str(int(P * 10000)) + '_' + diagnosis + '_' + file[:-4] ), dpi=100)
    else:
        plt.show()


