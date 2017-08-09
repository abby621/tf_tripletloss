import csv, random, os, glob
import numpy as np
import tensorflow as tf
from alexnet import CaffeNetPlaces365
import time
from dataretriever import DataRetriever

'''
If flipped files don't yet exist, run:

for file in /Users/abby/Documents/datasets/resized_traffickcam/*/*/*.jpg; do
  convert "$file" -flop "${file%.jpg}"_flipped.jpg
done

for file in /Users/abby/Documents/datasets/resized_expedia/*/*.jpg; do
  convert "$file" -flop "${file%.jpg}"_flipped.jpg
done
'''

traffickcam_im_file = '/Users/abby/Documents/datasets/resized_traffickcam/current_traffickcam_ims.txt'
with open(traffickcam_im_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    traffickcam_ims = list(rd)

traffickcam_ims.pop(0)

im_list = []
flipped_im_list = []
for im_id,hotel_id,im in traffickcam_ims:
    new_path = im.replace('/mnt/EI_Code/ei_code/django_ei/submissions/','/Users/abby/Documents/datasets/resized_traffickcam/')
    if os.path.exists(new_path):
        im_list.append((new_path,hotel_id))
    flipped_path = new_path.replace('.jpg','_flipped.jpg')
    if os.path.exists(flipped_path):
        flipped_im_list.append((flipped_path,hotel_id))

expedia_im_file = '/Users/abby/Documents/datasets/resized_expedia/current_expedia_ims.txt'
with open(expedia_im_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    expedia_ims = list(rd)

expedia_ims.pop(0)

for im_id,hotel_id,im in expedia_ims:
    new_path = os.path.join('/project/focus/datasets/traffickcam/expedia',str(hotel_id),str(im_id)+'.jpg')
    new_path = new_path.replace('/project/focus/datasets/traffickcam/expedia/','/Users/abby/Documents/datasets/resized_expedia/')
    if os.path.exists(new_path):
        im_list.append((new_path,hotel_id))
    flipped_path = new_path.replace('.jpg','_flipped.jpg')
    if os.path.exists(flipped_path):
        flipped_im_list.append((flipped_path,hotel_id))

ims_by_class = {}
for i in im_list:
    if not i[1] in ims_by_class:
        ims_by_class[i[1]] = {}
        ims_by_class[i[1]]['traffickcam'] = []
        ims_by_class[i[1]]['expedia'] = []
    if 'expedia' in i[0]: # expedia
        if not i[0] in ims_by_class[i[1]]['expedia']:
            ims_by_class[i[1]]['expedia'].append(i[0])
    else: # traffickcam
        if not i[0] in ims_by_class[i[1]]['traffickcam']:
            ims_by_class[i[1]]['traffickcam'].append(i[0])

# delete any classes that don't have both traffickcam and expedia images
classes = ims_by_class.keys()
numClassesStart = len(classes)
numIms = 0
for cls in classes:
    if len(ims_by_class[cls]['traffickcam']) == 0 or len(ims_by_class[cls]['expedia']) == 0:
        ims_by_class.pop(cls, None)
    else:
        numIms += len(ims_by_class[cls]['traffickcam'])
        numIms += len(ims_by_class[cls]['expedia'])

classes = ims_by_class.keys()
numClasses = len(classes)

allClasses = np.zeros((numIms),dtype='int')
allIms = []
tcOrExpedia = np.zeros((numIms),dtype='str')
startInd = 0
for cls in classes:
    for captureType in ['traffickcam','expedia']:
        allClasses[startInd:startInd+len(ims_by_class[cls][captureType])] = int(cls)
        tcOrExpedia[startInd:startInd+len(ims_by_class[cls][captureType])] = captureType[0]
        allIms.extend(ims_by_class[cls][captureType])
        startInd += len(ims_by_class[cls][captureType])

# RANDOM
img_size = [256, 256]
crop_size = [227, 227]
featLayer = 'fc7'
batch_size = 100

image_batch = tf.placeholder(tf.float32, shape=[None, crop_size[0], crop_size[0], 3])

print("Preparing network...")
net = CaffeNetPlaces365({'data': image_batch})
feat = net.layers[featLayer]

# tf will consume any GPU it finds on the system. Following lines restrict it to "first" GPU
c = tf.ConfigProto()
c.gpu_options.visible_device_list="0"

# Create data "batcher"
data = DataRetriever(im_list, img_size, crop_size, batch_size, False)

init_op = tf.global_variables_initializer()

print("Starting session...")
sess = tf.Session(config=c)
sess.run(init_op)
net.load('./models/places365/alexnet.npy', sess)
allFeats = np.zeros((len(data.files),net.layers['fc7'].shape[1].value))
num_iters = len(data.files) / batch_size
inds = range(0,len(data.files),batch_size)
num_iters = len(inds)
for start_ind in inds:
    start_time = time.time()
    batch = data.getBatch()
    f = sess.run(feat, feed_dict={image_batch: batch})
    allFeats[start_ind:start_ind+batch_size,:] = f
    duration = time.time() - start_time
    print('Step %d of %d: (%.3f sec)' % (start_ind/batch_size, num_iters, duration))

sess.close()

np.save('/project/focus/abby/tc_tripletloss/classes.npy',allClasses)
np.save('/project/focus/abby/tc_tripletloss/ims.npy',np.asarray(allIms))
np.save('/project/focus/abby/tc_tripletloss/feats.npy',allFeats)
np.save('/project/focus/abby/tc_tripletloss/classes_0_ind.npy',classes_0_ind)
np.save('/project/focus/abby/tc_tripletloss/traffickcamOrExpedia.npy',tcOrExpedia)

# =========== if we loaded from files, start here:
import csv
import numpy as np
import random
import os
import caffe
import cv2
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
from sklearn import preprocessing
import glob
import csv
import numpy as np
import random
import os

def getDist(feat,otherFeat):
    dist = (otherFeat - feat)**2
    dist = np.sum(dist)
    dist = np.sqrt(dist)
    return dist

allClasses = np.load('/project/focus/abby/tc_tripletloss/classes.npy')
allIms = np.load('/project/focus/abby/tc_tripletloss/ims.npy')
allFeats = np.load('/project/focus/abby/tc_tripletloss/feats.npy')
classes_0_ind = np.load('/project/focus/abby/tc_tripletloss/classes_0_ind.npy').item()
classes = classes_0_ind.keys()
tcOrExpedia = np.load('/project/focus/abby/tc_tripletloss/traffickcamOrExpedia.npy')

numClasses = len(classes)

doctoredIms = np.load('/project/focus/abby/tc_tripletloss/ims.npy')
doctored_expedia_files = np.asarray(glob.glob('/project/focus/datasets/traffickcam/doctored_expedia/*/*.jpg'))
doctored_traffickcam_files = np.asarray(glob.glob('/project/focus/datasets/traffickcam/doctored_traffickcam/*/*/*.jpg'))

doctored_ims = np.concatenate((doctored_expedia_files,doctored_traffickcam_files))

allIms_doctored = [s.replace('expedia','doctored_expedia') for s in allIms]
allIms_doctored = [s.replace('resized_traffickcam','doctored_traffickcam') for s in allIms_doctored]

allIms_doctored = np.asarray(allIms_doctored,dtype='S91')

doctored_im_inds = np.nonzero(np.in1d(allIms_doctored,doctored_ims))[0]

doctoredIms[doctored_im_inds] = allIms_doctored[doctored_im_inds]

# why did some of the gs in jpg disappear???
allIms_fixed = []
for ix in range(0,len(allIms)):
    if allIms[ix][-1] != 'g':
        allIms_fixed.append(allIms[ix]+'g')
    else:
        allIms_fixed.append(allIms[ix])

allIms = np.asarray(allIms_fixed)

setAsideClasses = []
setAsideClasses_0ind = []
setAsideQueries = []
setAsideDb = []
while len(setAsideClasses) < 500:
    cls = random.choice(classes)
    posInds = np.where(allClasses==int(cls))[0]
    if cls not in setAsideClasses and 't' in tcOrExpedia[posInds] and 'e' in tcOrExpedia[posInds]:
        class_0_ind = classes_0_ind[cls]
        setAsideClasses.append(cls)
        tcIms = posInds[np.where(tcOrExpedia[posInds]=='t')[0]]
        queryImInd = random.choice(tcIms)
        setAsideQueries.append((allIms[queryImInd],class_0_ind))
        exIms = posInds[np.where(tcOrExpedia[posInds]=='e')[0]]
        for imInd in random.sample(exIms,min(10,len(exIms))):
            setAsideDb.append((allIms[imInd],class_0_ind))

while len(setAsideClasses) < 1000:
    cls = random.choice(classes)
    if cls not in setAsideClasses and 'e' in tcOrExpedia[posInds]:
        class_0_ind = classes_0_ind[cls]
        setAsideClasses.append(cls)
        exIms = posInds[np.where(tcOrExpedia[posInds]=='e')[0]]
        for imInd in random.sample(exIms,min(10,len(exIms))):
            setAsideDb.append((allIms[imInd],class_0_ind))

with open('/project/focus/datasets/tc_tripletloss/true_test_queries_XL.txt','a') as true_test_queries:
    for im, hotel in setAsideQueries:
        true_test_queries.write('%s %s\n' % (im,hotel))

with open('/project/focus/datasets/tc_tripletloss/true_test_db_XL.txt','a') as true_test_db:
    for im, hotel in setAsideDb:
        true_test_db.write('%s %s\n' % (im,hotel))

allTriplets = []
numFeats = len(allFeats) # this is just for when we're sampling fewer than the whole set of features
distThresh = 13 # our examples must be less than this threshold
negativeMultiplier = 1.25 # but let our negative example be distThresh * negativeMultiplier away
numTriplets = 0
clsCtr = 0
clsSkippedCt = 0
for cls in classes:
    if cls not in setAsideClasses:
        print 'Class: ', cls
        print clsCtr, ' out of ', numClasses
        clsCtr += 1
        print 'total so far: ', numTriplets
        print 'on track for:', numTriplets/clsCtr*numClasses
        class_0_ind = classes_0_ind[cls]
        posInds = np.where(allClasses==int(cls))[0]
        negInds = np.where(allClasses!=int(cls))[0]
        if len(posInds) > 1:
            for ix in range(0,len(posInds)):
                anchorIm = doctoredIms[posInds[ix]]
                anchorFeat = allFeats[posInds[ix]]
                anchorType = tcOrExpedia[posInds[ix]]
                possiblePosInds = random.sample([posInds[aa] for aa in range(len(posInds))],min(50,len(posInds)))
                theseTriplets = []
                for posInd in possiblePosInds:
                    positiveIm = allIms[posInd]
                    if positiveIm != anchorIm:
                        positiveFeat = allFeats[posInd]
                        posDist = getDist(anchorFeat,positiveFeat)
                        if tcOrExpedia[posInds[aa]] != tcOrExpedia[posInds[ix]]:
                            minDistThresh = 5 # make sure we don't have almost exact duplicates as positive pairs
                        else:
                            minDistThresh = 10 # make sure if it's tc-tc or ex-ex, the positive pair isn't TOO close
                        if posDist < distThresh and posDist > minDistThresh:
                            negDist = 10000000
                            tries = 0
                            while negDist > distThresh and tries < 100:
                                numAdded = 0
                                while tries < 100 and numAdded < 8:
                                    tries += 1
                                    negInd = random.choice(negInds)
                                    negativeImClass = allClasses[negInd]
                                    while str(negativeImClass) in setAsideClasses:
                                        negInd = random.choice(negInds)
                                        negativeImClass = allClasses[negInd]
                                    negType = tcOrExpedia[negInd]
                                    negFeat = allFeats[negInd]
                                    negDist = getDist(anchorFeat,negFeat)
                                    if negDist <= posDist*negativeMultiplier:
                                        numTriplets += 1
                                        negativeIm = allIms[negInd]
                                        negativeImClass_0_ind = classes_0_ind[str(negativeImClass)]
    #                                     add this triplet to our list of all triplets
                                        theseTriplets.append((anchorIm, str(class_0_ind), positiveIm, str(class_0_ind), negativeIm, str(negativeImClass_0_ind),posDist,negDist))
                                        numAdded += 1
                allTriplets.extend(theseTriplets)
    else:
        clsSkippedCt += 1

randomOrder = range(len(allTriplets))
random.shuffle(randomOrder)
shuffledTriplets = [allTriplets[aa] for aa in randomOrder]

batchSize = 600
assert batchSize % 3 == 0

testTriplets = shuffledTriplets[:batchSize*4]
# grab everything else as train triplets, but make sure it's divisible by 3
trainTriplets = shuffledTriplets[batchSize*4+1:len(shuffledTriplets)-len(shuffledTriplets[batchSize*4+1:])%3]

smallTestTriplets = shuffledTriplets[:batchSize]
smallTrainTriplets = shuffledTriplets[batchSize+1:batchSize*2]

def write_triplet_file(triplets,batchSize,filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
    txtFile = open(filePath,'a')
    for ix in range(0,len(triplets),batchSize):
        anchorIms = []
        positiveIms = []
        negativeIms = []
        batch = triplets[ix:ix+batchSize/3]
        for triplet in batch:
            anchorIms.append((triplet[0],triplet[1]))
            positiveIms.append((triplet[2],triplet[3]))
            negativeIms.append((triplet[4],triplet[5]))
        for a in anchorIms:
            txtFile.write('%s %s\n' % (a[0],a[1]))
        for p in positiveIms:
            txtFile.write('%s %s\n' % (p[0],p[1]))
        for n in negativeIms:
            txtFile.write('%s %s\n' % (n[0],n[1]))
    txtFile.close()

write_triplet_file(trainTriplets,batchSize,'/project/focus/datasets/tc_tripletloss/train_triplets_tc_to_expedia_XL.txt')
write_triplet_file(testTriplets,batchSize,'/project/focus/datasets/tc_tripletloss/test_triplets_tc_to_expedia_XL.txt')
write_triplet_file(smallTrainTriplets,batchSize,'/project/focus/datasets/tc_tripletloss/small_train_triplets_tc_to_expedia_XL.txt')
write_triplet_file(smallTestTriplets,batchSize,'/project/focus/datasets/tc_tripletloss/small_test_triplets_tc_to_expedia_XL.txt')
