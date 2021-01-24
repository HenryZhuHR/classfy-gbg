import os

VOC_root = "../VOC2012"

voc_dir_list = [
    '/Annotations',
    '/ImageSets','/ImageSets/Main',
    '/JPEGImages'
]

for dir in voc_dir_list:
    dir_name=VOC_root+dir
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('-- Create  ',dir_name)
    else:
        print('-- Exists  ',dir_name)

print()
print("$ tree")
os.system("tree ../VOC2012")