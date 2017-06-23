

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import os

caffe.set_mode_gpu()

# SUB_W=544
# SUB_H=408

CLASS_NUM=41
# IMAGE_NUM=1481 #train data number
IMAGE_NUM=1461 #test data number

SUB_SEG=4
SUB_WIDTH=816
SUB_HEIGHT=612
SUB_W=272
SUB_H=204

def transform_img(img):
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
'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/jingyun/Documents/wood-deeplearning/data/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/jingyun/Documents/wood-deeplearning/models/model_2/caffenet_deploy_2.prototxt',
                '/home/jingyun/Documents/wood-deeplearning/models/model_2/caffe_model_2_iter_5000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../data/test/*JPG")]
# test_img_paths = [img_path for img_path in glob.glob("../data/train/*JPG")]

#Making predictions
test_ids = []
preds = []
Nrec=0
Nerr=0
for img_path in test_img_paths:
    print img_path
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    subimgs = transImgSub(img)
    counter=np.zeros(CLASS_NUM)
    for sub in subimgs:
        net.blobs['data'].data[...] = transformer.preprocess('data', sub)
        out = net.forward()
        pred_probas = out['prob']
        # print pred_probas.argmax()
        counter[pred_probas.argmax()]+=1
    max=np.array([0, counter[0]])
    for x in range(1, CLASS_NUM):
        temp=counter[x]
        if temp>max[1]:
            max[0]=x
            max[1]=temp
    baseName = os.path.basename(img_path)
    if (max[0]+1)==int(float(baseName[:2])):
        print 'TRUE'
        Nrec+=1
    else:
        print 'FALSE'
        Nerr+=1
    # print counter
    print str(Nrec)
    print str(Nerr)
    print '-------'

print 'recognition rate= '+ str(float(Nrec)/float(IMAGE_NUM))
print 'error rate= '+ str(float(Nerr)/float(IMAGE_NUM))