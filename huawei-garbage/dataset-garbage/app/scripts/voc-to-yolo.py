import os
import xml.etree.ElementTree as ET
import tqdm


VOC_root = "../VOC2012"

sets = ['train', 'val']
classes=[]
with open(VOC_root+'/ImageSets/Main'+'/classes.names','r') as f_class:
    classes=list(f_class)
for i in range(len(classes)):
        classes[i]=classes[i].replace('\n','')
print(classes)

# ---------------------------------------------
#   Convert VOC to YOLO
# ---------------------------------------------


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(VOC_root + '/Annotations/%s.xml' % (image_id))
    out_file = open(VOC_root + '/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')

for image_set in sets:
    if not os.path.exists(VOC_root + '/labels/'):
        os.makedirs(VOC_root + '/labels/')

    image_ids =[]
    with open(VOC_root + '/ImageSets/Main/%s.txt'%image_set,'r') as f_txt:
        image_ids=f_txt.read().split('\n')[:-1]
    
    
    for i in range(len(image_ids)):
        image_ids[i] = os.path.split(image_ids[i])[-1]
        image_ids[i] = image_ids[i].split('.')[0]
    list_file = open('%s.txt' % image_set, 'w')

    pbar=tqdm.tqdm(total=len(image_ids))
    for image_id in image_ids:
        list_file.write(VOC_root + '/JPEGImages/%s.jpg\n' % image_id)
        convert_annotation(image_id)
        pbar.update(1)
    list_file.close()
    pbar.close()
