import numpy as np 
import pandas as pd 
import os
import keras
import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models as sm
import warnings
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


sm.set_framework('tf.keras')
sm.framework()

# base_dir = 'G:/Major Project/Dataset'
# class_dict = pd.read_csv(os.path.join(base_dir, 'class_dict.csv'))
class_dict = pd.read_csv('class_dict.csv')

num_class = len(class_dict)



colors = []
for (r,g,b) in class_dict[['r', 'g', 'b']].to_numpy():
    colors.append([r,g,b])
map_color = {x:v for x,v in zip(range(num_class),colors)}


def onehot_to_rgb(onehot, color_dict= map_color):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)


backbones = ['resnet34', 'inceptionv3', 'vgg16']

def predict_image(model, img):
    if len(img.shape) == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    w, h = img.shape[:2]
    w_padded = (w // 256 + 1) * 256
    h_padded = (h // 256 + 1) * 256

    padding_shape = (w_padded, h_padded, 3)
    padded_img = np.pad(img, ((0, w_padded - img.shape[0]), (0, h_padded - img.shape[1]), (0, 0)), mode='constant')

    padded_img = padded_img / 255
    mask_shape = (w_padded, h_padded, 7)

    mask_padded = np.zeros(mask_shape)

    for i in range(0, mask_shape[0], 256):
        for j in range(0, mask_shape[1], 256):
            patch = padded_img[i:i + 256, j:j + 256, :]
            predicted = model.predict(np.expand_dims(patch, axis=0))
            mask_padded[i:i + 256, j:j + 256, :] = predicted

    return mask_padded[:img.shape[0], :img.shape[1], :]

# ... (rest of the code)


# ... rest of the code ...

models_path = ['resnet34.h5']

def predict(img, paths):
    predicted_masks = []
    for path in models_path:
        model = keras.models.load_model(path,
                                        custom_objects={'focal_loss_plus_jaccard_loss': sm.losses.categorical_focal_jaccard_loss,
                                                        'iou_score':sm.metrics.IOUScore, 
                                                        'threshold': 0.5,
                                                        'f1-score':sm.metrics.FScore})

        predicted_masks.append(predict_image(model, img))
    plt.figure(figsize=(15, 15))  # Set the figure size to 15x5 inches
    # plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
    plt.subplot(1, 1, 1)  # Create a subplot with 1 row and 2 columns, set current subplot to the first one


    # plt.subplot(152)
    plt.subplot(1, 1, 1)  # Set current subplot to the second one
    plt.title('Mask {}'.format(backbones[0]) ,fontsize=18)
    plt.axis('off')
    plt.imshow(onehot_to_rgb(predicted_masks[0]))
    plt.savefig('./static/models_masks.jpeg',dpi = 100)
    plt.close()  # Add this line after saving the figure

    return predicted_masks


