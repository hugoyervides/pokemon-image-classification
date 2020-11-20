import os
import cv2
from imutils import paths

def get_image_paths(path):
    return list(paths.list_images(path))


#Set the directory of the dataset in here!
DATASET_DIRECTORY = "archive/important_dataset"
OUTPUT_DATASET_FOLDER = "output"

classes = os.listdir(DATASET_DIRECTORY)

#Create the output folder
try:
    os.mkdir(OUTPUT_DATASET_FOLDER)
except:
    pass

os.chdir(OUTPUT_DATASET_FOLDER)

#Create the classes folders
for folder in classes:
    try:
        os.mkdir(folder)
    except:
        pass

os.chdir(os.path.dirname(os.getcwd()))

for x in classes:
    images = get_image_paths(DATASET_DIRECTORY + '/' + x)
    count = 0
    for image_path in images:
        image = cv2.imread(image_path)
        cv2.imwrite( OUTPUT_DATASET_FOLDER + '/' + x + '/' + str(count) + '.png', image)
        count += 1