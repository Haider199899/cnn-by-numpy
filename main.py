import os.path
from src.dataloader import DataLoader

data_dir = os.path.abspath('./data/all')
train_dir = os.path.abspath('./data/train')
valid_dir = os.path.abspath('./data/val')
test_dir = os.path.abspath('./data/test')
image_size = (32,32)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
dl = DataLoader(data_dir, train_dir, valid_dir, test_dir, image_size, train_ratio,val_ratio, test_ratio)
data = dl.load_data()
dl.split_data(data)
train_data,train_labels = dl.convert_img_to_array(train_dir,image_size)
train_data = dl.normalize_images(train_data)
test_data,test_labels = dl.convert_img_to_array(test_dir,image_size)
test_data = dl.normalize_images(test_data)
valid_data,valid_labels = dl.convert_img_to_array(valid_dir,image_size)
valid_data = dl.normalize_images(valid_data)
