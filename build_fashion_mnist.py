import csv
import os
import shutil

import cv2
from keras.datasets import fashion_mnist

LABEL_MAPPING = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]
OUTPUT_PATH = 'fashion_mnist'
TRAIN_PATH = os.path.sep.join([OUTPUT_PATH, 'train'])
TEST_PATH = os.path.sep.join([OUTPUT_PATH, 'test'])

if os.path.isdir(OUTPUT_PATH):
    print('[WARN] Output path exists. Deleting it.')
    shutil.rmtree(OUTPUT_PATH)

print('[INFO] Creating output directories.')
os.makedirs(OUTPUT_PATH)
os.makedirs(TRAIN_PATH)
os.makedirs(TEST_PATH)

print('[INFO] Loading Fashion-MNIST.')
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


def write_to_disk(csv_path, images_path, X, y):
    with open(csv_path, 'w') as f:
        field_names = ['image_path', 'label']
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()

        for index, (image, label) in enumerate(zip(X, y)):
            label_path = os.path.sep.join([images_path, str(label)])

            if not os.path.exists(label_path):
                os.makedirs(label_path)

            image_path = os.path.sep.join([label_path, f'{index}.png'])
            cv2.imwrite(image_path, image)

            writer.writerow({'image_path': image_path, 'label': LABEL_MAPPING[int(label)]})


if __name__ == '__main__':
    print('[INFO] Writing training files to disk.')
    TRAIN_CSV_PATH = os.path.sep.join(['.', 'fashion_mnist_train.csv'])
    write_to_disk(TRAIN_CSV_PATH, TRAIN_PATH, X_train, y_train)

    print('[INFO] Writing testing files to disk.')
    TEST_CSV_PATH = os.path.sep.join(['.', 'fashion_mnist_test.csv'])
    write_to_disk(TEST_CSV_PATH, TEST_PATH, X_test, y_test)
