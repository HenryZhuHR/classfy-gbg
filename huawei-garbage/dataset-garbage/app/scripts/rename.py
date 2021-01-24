import os
from pprint import pprint
import re

img_dir = "../images"


for dir_content in os.walk(img_dir):
    # --- dir_content ---
    # ('../images', ['parking'], [])
    # ('../images/parking', [], ['a07a35761c8b066589b60d1447dd7b0d.jpeg',...)
    # -------------------
    #   dir_content[0] = root
    #   dir_content[1] = dirs list
    #   dir_content[2] = files list
    
    pprint(dir_content[2])
    if not (len(dir_content[2])>0):     # no file in dir
        continue
    
    root=dir_content[0]
    files=dir_content[2]  
    className=os.path.split(root)[-1]

    # filter all images
    imgs=[]     # need to be renamed
    n_renamed_img=0
    print("====== Matched List ======")
    for file in files:  
        name=file.split('.')[0]
        suffixes=file.split('.')[-1]
        if suffixes in ['jpg','jpeg']:      # is image file
            namesplit=name.split('-')
            nameMatch=(namesplit[0]==className)and(namesplit[1].isalnum())
            if nameMatch:   # match, has been renamed
                n_renamed_img+=1
                print(root+'/'+file,'\t',nameMatch)
            else:           # no match, has not been renamed
                imgs.append(file)

    nimg=len(imgs)
    print(" -- Class: ",className, len(imgs),'\t',root)

    print("====== to be Matched List ======")
    for img in imgs:
        dstName=className+"-"+str("%05d.jpg"%n_renamed_img)
        print(img,'\t',dstName)
        print(root)
        os.rename(root+'/'+img,root+'/'+dstName)
        n_renamed_img+=1
