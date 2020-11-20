from random import shuffle
import glob
import numpy as np
import h5py
import os

#CONSTANTS
OUTPUT_FILE = 'pikachu_charmander.hdf5'  # OUTPUT H5 FILE
ROOT_DIRECTORY = 'output'

#TRAIN, TEST DATA
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
DEV_SIZE = 0.1

#Function to shuffle the data
def shuffle_data(addrs, labels):
    c = list(zip(addrs, labels))
    shuffle(c)
    (addrs, labels) = zip(*c)
    return (addrs, labels)

#Function to get the path to all images inside the directory
def get_image_addrs(directory_path):
    return glob.glob(directory_path)

#Function to build the list of classes
def build_classes(dataset_root_directory):
    return [ name for name in os.listdir(dataset_root_directory) if os.path.isdir(os.path.join(dataset_root_directory, name)) ]

#Function to parse the classes to numerical IDs
def parse_classes_to_ids(classes):
    ids = {}
    cont = 0
    for i in classes:
        ids[i] = cont
        cont += 1
    return ids

#Function to build labels array
def build_labels_array(addrs, image_class_id):
    labels = []
    for address in addrs: 
        labels.append(image_class_id)
    return labels

#function to divide the data
def divide_data(addrs, labels):
    train_addrs = addrs[0: int(TRAIN_SIZE * len(addrs))]
    train_labels = labels[0: int(TRAIN_SIZE * len(labels))]
    test_addrs = addrs[int(TRAIN_SIZE * len(addrs)):]
    test_labels = labels[int(TRAIN_SIZE * len(labels)):]
    return (train_addrs, train_labels, test_addrs, test_labels)

#Init arrays
train_addrs = []
train_labels = []
test_addrs = []
test_labels = []

#Bild the classes and IDs
classes = build_classes(ROOT_DIRECTORY)
ids = parse_classes_to_ids(classes)

#Loop thru the classes
for image_class in classes:
    #get image paths
    addrs = get_image_addrs(ROOT_DIRECTORY + '/' + image_class + '/*.png')
    labels = build_labels_array(addrs, ids[image_class])
    #Shuffle the data
    (addrs, labels) = shuffle_data(addrs, labels)
    #Divide the data 
    (train_addrs_class, train_labels_class, test_addrs_class, test_labels_class) = divide_data(addrs, labels)                               
    #Append the data to the existing arrays
    train_addrs.extend(train_addrs_class)
    train_labels.extend(train_labels_class)
    test_addrs.extend(test_addrs_class)
    test_labels.extend(test_labels_class)


train_shape = (len(train_addrs), 128, 128, 3)
test_shape = (len(test_addrs), 128, 128, 3)

# open a hdf5 file and create earrays 
f = h5py.File(OUTPUT_FILE, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f.create_dataset("train_img", train_shape, np.uint8)
f.create_dataset("test_img", test_shape, np.uint8)  

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key. 
f.create_dataset("train_labels", (len(train_addrs),), np.uint8)
f["train_labels"][...] = train_labels

f.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f["test_labels"][...] = test_labels

f.create_dataset("list_classes", data=np.array(classes, 'S7'))
######################## third part: write the images #########################
import cv2

# loop over train paths
for i in range(len(train_addrs)):
  
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)) )

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)# resize to (128,128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
    f["train_img"][i, ...] = img[None] 

# loop over test paths
for i in range(len(test_addrs)):

    if i % 1000 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)) )

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["test_img"][i, ...] = img[None]

f.close()