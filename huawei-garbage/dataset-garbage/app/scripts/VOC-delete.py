VOC_root = "../VOC2012/"

import shutil
str = input("input  delete/cancel  to finish the operate: ")

if (str=="delete"):
    shutil.rmtree(path=VOC_root)
elif(str=="cancel"):
    pass
else:
    print("-- Invaild Operate!!")
    print("-- Please try again input:  delete or cancel")