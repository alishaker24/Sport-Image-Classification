import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.model_selection import train_test_split
import pandas as pd
#import tensorflow.keras as keras

tf.config.list_physical_devices('GPU')
    
OLD_DIR = '/kaggle/input/nn23-sports-image-classification/Test'
TRAIN_DIR = 'NewTrain'
TEST_DIR = 'Test'
IMG_SIZE = 100
LR = 0.001

MODEL_NAME = 'CNN'

# def augmentation():
#     data_augmentation = keras.Sequential([
#         keras.layers.RandomFlip("horizontal"),
#         keras.layers.RandomRotation(20),
#         keras.layers.RandomZoom((-0.2,0.2)),
#         keras.layers.RandomContrast(0.1),  
#         keras.layers.RandomTranslation(0.1,0.1)])


#     data_augmentation.compile

#     return data_augmentation


# def png_to_jpeg_converter(OLD_DIR):
#       base_path = OLD_DIR
#       new_path = 'New Train'
#       for infile in os.listdir(TRAIN_DIR):
#           print("file : " + infile)
#           read = cv2.imread(TRAIN_DIR + '/' + infile)
#           outfile = infile.split('.')[0] + '.jpg'
#           cv2.imwrite(outfile, read, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-2]
    #print(word_label)

    if word_label[0:1] == 'B':
        return np.array([1,0,0,0,0,0])
    elif word_label[0:1] == 'F':
        return np.array([0,1,0,0,0,0])
    elif word_label[0:1] == 'R':
        return np.array([0,0,1,0,0,0])
    elif word_label[0:1] == 'S':
        return np.array([0,0,0,1,0,0])
    elif word_label[0:1] == 'T':
        return np.array([0,0,0,0,1,0])
    elif word_label[0:1]  == 'Y':
        return np.array([0,0,0,0,0,1])

    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data



def create_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    train_data =np.load('train_data.npy',allow_pickle=True)
   
else: # If dataset is not created:
    train_data = create_train_data()

#   print(train_data.shape)
if (os.path.exists('test_data.npy')):
    test_data =np.load('test_data.npy',allow_pickle=True)
else:
    test_data = create_test_data()

train = train_data
test = test_data


#inputs & outputs
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = np.array([i[1] for i in train])

# data_augmentation = keras.Sequential([
# keras.layers.RandomFlip("horizontal"),
# keras.layers.RandomRotation(20),
# keras.layers.RandomZoom((-0.2,0.2)),
# keras.layers.RandomContrast(0.1),  
# keras.layers.RandomTranslation(0.1,0.1)])

# data_augmentation.compile

print (X.shape)
print (Y.shape)
X_train,X_v_test, Y_train, Y_v_test = train_test_split(X, Y, test_size=0.32, random_state=117)

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = np.array([i[1] for i in test])

tf.compat.v1.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 3, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 3, activation='relu')
pool5 = max_pool_2d(conv5, 5)

conv6 = conv_2d(pool5, 512, 5, activation='relu')
pool6 = max_pool_2d(conv6, 5)

fully_layer = fully_connected(pool6, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
print (X_train.shape)

if (os.path.exists('model.tfl.meta')):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train}, batch_size=16, n_epoch=50,
          validation_set=({'input': X_v_test}, {'targets': Y_v_test}),
          snapshot_step=440, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

validate=model.evaluate(X_v_test,Y_v_test)
print('evaluate')
print(validate)



tst = pd.read_csv("Submit.csv")

pred=[]
name=[]

maxval=0

for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, img)
    test_img = cv2.imread(path, cv2.IMREAD_COLOR) # [0,1,2] = [b,g,r]
    test_img = test_img[:, :, [2, 1, 0]]      
    test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict([test_img])[0]
    name.append(img)
    maxval=max(prediction)
    for i in range( len(prediction)):
       
        if prediction[i] == maxval:
            indx = i 
    pred.append(indx)


sub = tst[["image_name"]]
sub ["image_name"] = name
sub["label"]= pred
sub.to_csv("CNN7.csv")
# plt.show()

#delete
if os.path.exists("checkpoint"):
  os.remove("checkpoint")
else:
  print("The file does not exist")
 
if os.path.exists("model.tfl.data-00000-of-00001"):
  os.remove("model.tfl.data-00000-of-00001")
else:
  print("The file does not exist")    

if os.path.exists("model.tfl.index"):
  os.remove("model.tfl.index")
else:
  print("The file does not exist") 
  
if os.path.exists("model.tfl.meta"):
  os.remove("model.tfl.meta")
else:
  print("The file does not exist") 

if os.path.exists("test_data.npy"):
  os.remove("test_data.npy")
else:
  print("The file does not exist") 

if os.path.exists("train_data.npy"):
  os.remove("train_data.npy")
else:
  print("The file does not exist")
