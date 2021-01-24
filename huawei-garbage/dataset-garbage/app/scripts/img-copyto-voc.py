import os
import re
import shutil
import pprint
from tqdm import tqdm
import time

img_dir = '../images'
voc_dir = '../VOC2012'


for dir_content in os.walk(img_dir):
    root = dir_content[0]
    files = dir_content[2]
    nfiles = len(files)
    className = root.split('/')[-1]

    if not nfiles > 0:
        continue    

    nimgs=0
    for file in files:
        suffixes = re.search("[^\.]\w*$", file).group()
        if suffixes in ['jpg', 'jpeg']:      # is image file
            nimgs+=1
    
    pbar = tqdm(total=nimgs)
    print(" --- Class\t%s" % className)
    for file in files:
        suffixes = re.search("[^\.]\w*$", file).group()
        if suffixes in ['jpg', 'jpeg']:      # is image file
            if os.path.exists(root+'/'+file.split('.')[0]+'.xml'):
                srcpath=root+'/'+file.split('.')[0]+'.jpg'
                dstpath=voc_dir+'/'+'JPEGImages/'+file.split('.')[0]+'.jpg'
                # print(srcpath,dstpath)
                shutil.copyfile(srcpath,dstpath)

                srcpath=root+'/'+file.split('.')[0]+'.xml'
                dstpath=voc_dir+'/'+'Annotations/'+file.split('.')[0]+'.xml'
                # print(srcpath,dstpath)
                shutil.copyfile(srcpath,dstpath)

                pbar.update(1)
                time.sleep(0.01)


