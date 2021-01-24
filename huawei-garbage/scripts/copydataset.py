import os
import shutil
from tqdm import tqdm

# Copy dataset 

src_dir="dataset-garbage/images"
dst_dir="data"

# divide dataset into train&val
traindir = os.path.join(dst_dir, 'train')
valdir = os.path.join(dst_dir, 'val')

train_percent=0.8
val_percent=1-train_percent