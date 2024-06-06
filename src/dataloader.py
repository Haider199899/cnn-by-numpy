import shutil
import os
from PIL import Image
import numpy as np
class DataLoader:
    def __init__(self, data_dir,train_dir,val_dir,test_dir, image_size, train_ratio, val_ratio, test_ratio):
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio


    def load_data(self):
        print('Loading data...')
        each_class_dir = [x[0] for x in os.walk(self.data_dir)]
        each_class_dir.pop(0)
        data = {}
        size = 0
        for dir in each_class_dir:
            label = dir.split('/')[-1]
            data[label] = []
            for file in os.listdir(dir):
                x = os.path.abspath(os.path.join(dir, file))
                data[label].append(x)
            size += len(data[label])
        print('Total number of images : ',size)
        return data

    def split_data(self,data):
        print('Splitting data...')
        labels = data.keys()
        total_training_images = 0
        total_validation_images = 0
        total_testing_images = 0

        # Making directory of each label
        for label in labels:
            if(os.path.exists(os.path.join(self.train_dir,label)) == False):
               os.mkdir(os.path.join(self.train_dir,label))
            train_label_folder = os.path.abspath(os.path.join(self.train_dir,label))
            files = data[label]
            train_size = int(len(files) * self.train_ratio)
            train_images = files[:train_size]
            total_training_images += len(train_images)
            self.move_files(train_label_folder,train_images)

            if (os.path.exists(os.path.join(self.val_dir, label)) == False):
                os.mkdir(os.path.join(self.val_dir, label))
            val_label_folder = os.path.abspath(os.path.join(self.val_dir, label))
            files = data[label]
            val_size = int(len(files) * self.val_ratio)
            val_images = files[train_size:train_size + val_size]
            total_validation_images += len(val_images)
            self.move_files(val_label_folder,val_images)

            if (os.path.exists(os.path.join(self.test_dir, label)) == False):
                os.mkdir(os.path.join(self.test_dir, label))
            test_label_folder = os.path.abspath(os.path.join(self.test_dir, label))
            files = data[label]
            test_images = files[train_size + val_size:]
            total_testing_images += len(test_images)
            self.move_files(test_label_folder,test_images)

    def convert_img_to_array(self,dir,image_size):
        print('Converting images to array with fixed size....')
        data = []
        labels = []
        classes = os.listdir(dir)
        classes.remove('.DS_Store')
        for class_idx, class_name in enumerate(classes):
            img_dir = os.path.join(dir, class_name)
            for img in os.listdir(img_dir):
                image_path = os.path.join(img_dir, img)
                img = Image.open(image_path).convert('RGB').resize(image_size)
                np_img = np.array(img)
                data.append(np_img)
                labels.append(class_idx)
        return np.array(data), np.array(labels)

    def normalize_images(self,images):
        return images / 255.0

    def move_files(self,dest_folder,files):
        for file in files:
            shutil.move(file,dest_folder)


