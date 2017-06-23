
import os
import glob
import random
import numpy as np

import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

SUB_SEG=4
SUB_NUM=16
SUB_WIDTH=816
SUB_HEIGHT=612
SUB_W=227
SUB_H=227


def transformImg(img):
    img[:, :, 0]=cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1]=cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2]=cv2.equalizeHist(img[:, :, 2])
    img = cv2.resize(img, (SUB_W, SUB_H), interpolation=cv2.INTER_CUBIC)

    return img

def transImgSub(img):
    # img[:, :, 0]=cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 1]=cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 2]=cv2.equalizeHist(img[:, :, 2])
    subImgs=[]
    for i in range(SUB_SEG):
        for j in range(SUB_SEG):
            subImg=img[SUB_HEIGHT*j : SUB_HEIGHT*(j+1), SUB_WIDTH*i : SUB_WIDTH*(i+1)]
            subImg=cv2.resize(subImg, (SUB_W, SUB_H), interpolation=cv2.INTER_CUBIC)
            subImgs.append(subImg)
    return subImgs
# img=cv2.imread('../data/0128.JPG', cv2.IMREAD_COLOR)
# subimages=transImgSub(img)
# for sub in subimages:
#     cv2.imshow("sub", sub)
#     cv2.waitKey()

def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=3,
        width=SUB_W,
        height=SUB_H,
        label=label,
        data=np.rollaxis(img, 2).tostring()
    )

train_lmdb='../data/train_lmdb'
validation_lmdb='../data/validation_lmdb'

os.system('rm -rf '+train_lmdb)
os.system('rm -rf '+validation_lmdb)

train_data=[img for img in glob.glob('../data/train/*JPG')]
#tset_data=[img for img in glob.glob('../data/test/*JPG')]

random.shuffle(train_data)

print ('Creating train_lmdb')
in_db=lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for index, imgPath in enumerate(train_data):
        if index%6==0:
            continue
        img=cv2.imread(imgPath, cv2.IMREAD_COLOR)
        subimgs=transImgSub(img)
        #img=transformImg(img)
        baseName = os.path.basename(imgPath)
        if baseName.startswith('01'):
            label = 0
        elif baseName.startswith('02'):
            label = 1
        elif baseName.startswith('03'):
            label = 2
        elif baseName.startswith('04'):
            label = 3
        elif baseName.startswith('05'):
            label = 4
        elif baseName.startswith('06'):
            label = 5
        elif baseName.startswith('07'):
            label = 6
        elif baseName.startswith('08'):
            label = 7
        elif baseName.startswith('09'):
            label = 8
        elif baseName.startswith('10'):
            label = 9
        elif baseName.startswith('11'):
            label = 10
        elif baseName.startswith('12'):
            label = 11
        elif baseName.startswith('13'):
            label = 12
        elif baseName.startswith('14'):
            label = 13
        elif baseName.startswith('15'):
            label = 14
        elif baseName.startswith('16'):
            label = 15
        elif baseName.startswith('17'):
            label = 16
        elif baseName.startswith('18'):
            label = 17
        elif baseName.startswith('19'):
            label = 18
        elif baseName.startswith('20'):
            label = 19
        elif baseName.startswith('21'):
            label = 20
        elif baseName.startswith('22'):
            label = 21
        elif baseName.startswith('23'):
            label = 22
        elif baseName.startswith('24'):
            label = 23
        elif baseName.startswith('25'):
            label = 24
        elif baseName.startswith('26'):
            label = 25
        elif baseName.startswith('27'):
            label = 26
        elif baseName.startswith('28'):
            label = 27
        elif baseName.startswith('29'):
            label = 28
        elif baseName.startswith('30'):
            label = 29
        elif baseName.startswith('31'):
            label = 30
        elif baseName.startswith('32'):
            label = 31
        elif baseName.startswith('33'):
            label = 32
        elif baseName.startswith('34'):
            label = 33
        elif baseName.startswith('35'):
            label = 34
        elif baseName.startswith('36'):
            label = 35
        elif baseName.startswith('37'):
            label = 36
        elif baseName.startswith('38'):
            label = 37
        elif baseName.startswith('39'):
            label = 38
        elif baseName.startswith('40'):
            label = 39
        else:
            label = 40
        for x in range(SUB_NUM):
            datum=make_datum(subimgs[x], label)
            num=(index)*SUB_NUM+x+1
            in_txn.put('{:0>5d}'.format(num), datum.SerializeToString())
            print ('{:0>5d}'.format(num) + ':' + imgPath)
        #datum = make_datum(img, label)
        #in_txn.put('{:0>3d}'.format(index), datum.SerializeToString())
        #print ('{:0>4d}'.format(index) + ':' + imgPath)
        #print '{:0>3d}'.format(index) + ':' + str(label)
in_db.close()


print ('\nCreating validation_lmdb')
in_db=lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for index, imgPath in enumerate(train_data):
        if index%6!=0:
            continue
        img=cv2.imread(imgPath, cv2.IMREAD_COLOR)
        subimgs = transImgSub(img)
        #img = transformImg(img)
        baseName = os.path.basename(imgPath)
        if baseName.startswith('01'):
            label = 0
        elif baseName.startswith('02'):
            label = 1
        elif baseName.startswith('03'):
            label = 2
        elif baseName.startswith('04'):
            label = 3
        elif baseName.startswith('05'):
            label = 4
        elif baseName.startswith('06'):
            label = 5
        elif baseName.startswith('07'):
            label = 6
        elif baseName.startswith('08'):
            label = 7
        elif baseName.startswith('09'):
            label = 8
        elif baseName.startswith('10'):
            label = 9
        elif baseName.startswith('11'):
            label = 10
        elif baseName.startswith('12'):
            label = 11
        elif baseName.startswith('13'):
            label = 12
        elif baseName.startswith('14'):
            label = 13
        elif baseName.startswith('15'):
            label = 14
        elif baseName.startswith('16'):
            label = 15
        elif baseName.startswith('17'):
            label = 16
        elif baseName.startswith('18'):
            label = 17
        elif baseName.startswith('19'):
            label = 18
        elif baseName.startswith('20'):
            label = 19
        elif baseName.startswith('21'):
            label = 20
        elif baseName.startswith('22'):
            label = 21
        elif baseName.startswith('23'):
            label = 22
        elif baseName.startswith('24'):
            label = 23
        elif baseName.startswith('25'):
            label = 24
        elif baseName.startswith('26'):
            label = 25
        elif baseName.startswith('27'):
            label = 26
        elif baseName.startswith('28'):
            label = 27
        elif baseName.startswith('29'):
            label = 28
        elif baseName.startswith('30'):
            label = 29
        elif baseName.startswith('31'):
            label = 30
        elif baseName.startswith('32'):
            label = 31
        elif baseName.startswith('33'):
            label = 32
        elif baseName.startswith('34'):
            label = 33
        elif baseName.startswith('35'):
            label = 34
        elif baseName.startswith('36'):
            label = 35
        elif baseName.startswith('37'):
            label = 36
        elif baseName.startswith('38'):
            label = 37
        elif baseName.startswith('39'):
            label = 38
        elif baseName.startswith('40'):
            label = 39
        else:
            label = 40
        for x in range(SUB_NUM):
            datum=make_datum(subimgs[x], label)
            num=(index)*SUB_NUM+x+1
            in_txn.put('{:0>5d}'.format(num), datum.SerializeToString())
            print ('{:0>5d}'.format(num) + ':' + imgPath)
        #datum = make_datum(img, label)
        #in_txn.put('{:0>3d}'.format(index), datum.SerializeToString())
        # print( '{:0>4d}'.format(index) + ':' + imgPath)
in_db.close()

print ('\nFinished processing alll images')




