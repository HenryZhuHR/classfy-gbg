import os
import shutil
import pprint
from tqdm import tqdm


VOC_root = "../VOC2012"

voc_dir_list = [
    '/Annotations',
    '/ImageSets','/ImageSets/Main',
    '/JPEGImages'
]
# ---------------------------------------------
#   Generate classes txt
# ---------------------------------------------
classes=[]
for dir_content in os.walk(VOC_root+'/JPEGImages'):
    # ('../VOC2012/JPEGImages', [], ['parking-00061.jpg', ...)

    root=dir_content[0] # '../VOC2012/JPEGImages'
    files=dir_content[2]

    pbar = tqdm(total=len(files))
    for file in files:        
        classname=file.split('-')[0]        
        if classname not in classes:
            classes.append(classname)
        pbar.update(1)

print(classes,VOC_root+'/ImageSets/Main'+'/classes.names')

with open(VOC_root+'/ImageSets/Main'+'/classes.names','w') as f_class:
    for classname in classes:
        f_class.write(classname+'\n')

# ---------------------------------------------
#   txt
# ---------------------------------------------
for dir_content in os.walk(VOC_root+'/JPEGImages'):
    # ('../VOC2012/JPEGImages', [], ['parking-00061.jpg', ...)

    root=dir_content[0] # '../VOC2012/JPEGImages'
    files=dir_content[2]

    for class_ in classes:
        print(' --- Class\t%s'%class_)
        with open(VOC_root +'/ImageSets/Main/%s.txt' % class_,'w') as f_txt:
            for file in files:
                if file.split('-')[0]==class_:
                    content=os.path.abspath(root+'/'+file)
                    # print(file,class_,content)
                    f_txt.write(content+"\n")

train_percent=0.8
val_percent=1-train_percent

if os.path.exists(VOC_root +'/ImageSets/Main/train.txt'):
    os.remove(VOC_root +'/ImageSets/Main/train.txt')
f_traintxt=open(VOC_root +'/ImageSets/Main/train.txt','a')

if os.path.exists(VOC_root +'/ImageSets/Main/val.txt'):
    os.remove(VOC_root +'/ImageSets/Main/val.txt')
f_valtxt=open(VOC_root +'/ImageSets/Main/val.txt','a')

for class_ in classes:       
    f_txt= open(VOC_root +'/ImageSets/Main/%s.txt' % class_,'r')
    files=list(f_txt)
    f_txt.close()
    for i in range(len(files)):
        files[i]=files[i].replace('\n','')
    print(files[:1])

    train_num=int(len(files)*0.8)
    train_list=files[:train_num]
    for content in train_list:
        f_traintxt.write(content+'\n')

    val_list=files[train_num:]
    for content in val_list:
        f_valtxt.write(content+'\n')

    # print("train:%5d\tval:%5d"%(len(train_list),len(val_list)))
f_traintxt.close()
f_valtxt.close()


