import os.path
from src.dataloader import DataLoader

data_dir = os.path.abspath('./data/all')
train_dir = os.path.abspath('./data/train')
valid_dir = os.path.abspath('./data/val')
test_dir = os.path.abspath('./data/test')
image_size = 320
train_ratio = 0.8
val_ratio = 0.9
test_ratio = 0.9
dl = DataLoader(data_dir, train_dir, valid_dir, test_dir, image_size, train_ratio,val_ratio, test_ratio)
data = dl.load_data()
dl.split_data(data)