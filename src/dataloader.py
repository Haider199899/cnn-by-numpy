import shutil
import os
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
        train_size = 0
        val_size = 0
        test_size = 0
        # Making directory of each label
        for label in labels:
            if(os.path.exists(os.path.join(self.train_dir,label)) == False):
               os.mkdir(os.path.join(self.train_dir,label))
               train_label_folder = os.path.abspath(os.path.join(self.train_dir,label))
               files = data[label]
               train_images = files[:int(len(files) * self.train_ratio)]
               train_size += len(train_images)
               self.move_files(train_label_folder,train_images)

            if (os.path.exists(os.path.join(self.val_dir, label)) == False):
                os.mkdir(os.path.join(self.val_dir, label))
                val_label_folder = os.path.abspath(os.path.join(self.val_dir, label))
                files = data[label]
                val_images = files[int(len(files) * self.train_ratio) : int(len(files) * self.val_ratio)]
                val_size += len(val_images)
                self.move_files(val_label_folder,val_images)

            if (os.path.exists(os.path.join(self.test_dir, label)) == False):
                os.mkdir(os.path.join(self.test_dir, label))
                test_label_folder = os.path.abspath(os.path.join(self.test_dir, label))
                files = data[label]
                test_images = files[int(len(files) * self.val_ratio): int(len(files) * self.test_ratio)]
                test_size += len(test_images)
                self.move_files(test_label_folder,test_images)
        print('Training images: ',len(train_images))
        print('Validation images: ',len(val_images))
        print('Test images: ',len(test_images))




    def preprocess_data(self):
        pass

    def move_files(self,dest_folder,files):
        for file in files:
            shutil.move(file,dest_folder)


