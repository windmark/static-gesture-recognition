import math
import json
import getPositions
import numpy as np

def calculateDistances(rightPalmPosition, rightFingerTipPositions, leftPalmPosition, leftFingerTipPositions):
    featureVector = []
    if rightPalmPosition != [] and rightFingerTipPositions != []:
        for n in range(0, len(rightFingerTipPositions)):
            # Extract the distance in each direction
            x = abs(rightFingerTipPositions[n][0] - rightPalmPosition[0])
            y = abs(rightFingerTipPositions[n][2] - rightPalmPosition[2])

            # Add the euclidian distance to the feature vector
            fingerDistance = euclidianDistance(x, y)

            featureVector.append(euclidianDistance(x,y))

    if (leftPalmPosition != [] and leftFingerTipPositions != []):
        for n in range(0, len(leftFingerTipPositions)):
            # Extract the distance in each direction
            x = abs(leftFingerTipPositions[n][0] - leftPalmPosition[0])
            y = abs(leftFingerTipPositions[n][2] - leftPalmPosition[2])

            # Add the euclidian distance to the feature vector
            featureVector.append(euclidianDistance(x,y))
    return featureVector

def euclidian3DDistance(x,y,z):
    return math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))

def euclidianDistance(x,y):
    return math.sqrt(math.pow(x,2) + math.pow(y,2))

'''
Normalizes and rounds to five decimals
'''
def normalize(array):
    normalizedData = []
    for n in range(0,2):
        minval = min(array[n*5:(n+1)*5])
        maxval= max(array[n*5:(n+1)*5])

        for a in range(n * 5, (n+1) * 5):
            tmp = array[a]
            #rounding to 5 decimals
            normalizedData.append(round((tmp - minval) / (maxval - minval),5))
        
    return normalizedData

'''
Parsing the txt files to extract the data.
'''
def readRawData(file):
    labelData = np.loadtxt(file, delimiter = '\t', usecols = (0,), dtype = str, unpack = True)
    rawLeftPalmData = np.loadtxt(file, delimiter = '\t', usecols = (1,), dtype = str)
    rawRightPalmData = np.loadtxt(file, delimiter = '\t', usecols = (2,), dtype = str)
    rawFingerData = np.loadtxt(file, delimiter = '\t', usecols = (3,), dtype = str)

    leftPalmData = []
    for item in rawLeftPalmData:
        item = item.translate(None, '[],')
        floats = [float(s) for s in item.split()]
        leftPalmData.append(floats)

    rightPalmData = []
    for item in rawRightPalmData:
        item = item.translate(None, '[],')
        floats = [float(s) for s in item.split()]
        rightPalmData.append(floats)

    fingerData = []
    for item in rawFingerData:
        item = item.translate(None, '[],')
        itemArray = [float(s) for s in item.split(" ")]

        i = 1
        temp3 = []
        temp30 = []
        for element in itemArray:
            temp3.append(element)
            if i % 3 == 0:
                temp30.append(temp3)
                temp3 = []

            if i % 30 == 0:
                fingerData.append(temp30)
                break
            i += 1
    return (leftPalmData, rightPalmData, fingerData, labelData)


'''
Orders data to featureVector format: right palm, right fingers, left palm, left fingers
Normalizing
'''
def convertToFeatureVectors(leftPalmData, rightPalmData, fingerData):
    featureVectorList = []
    for i in range(0, len(leftPalmData)):
        leftPalm = leftPalmData[i]
        rightPalm = rightPalmData[i]
        fingers = fingerData[i]

        half = len(fingers) / 2
        featureVector = calculateDistances(rightPalm, fingers[half:], leftPalm, fingers[:half])
        featureVectorList.append(normalize(featureVector))
    return(featureVectorList)


'''
Processes the raw data into feature files, which can later be used to train the models.
It expects featureFileName to be the path to a file where to save processed features,
based on the file dataFileName.
'''
def saveFeatureData(featureFileName, dataFileName):
    (leftPalmData, rightPalmData, fingerData, labelData) = readRawData(dataFileName)
    featureVectorList = convertToFeatureVectors(leftPalmData, rightPalmData, fingerData)

    featureFile = open('training/' + featureFileName, 'a')
    i = 0
    for vector in featureVectorList:
        featureFile.write("{}".format(labelData[i]))
        for feature in vector:
            featureFile.write(", {}".format(feature))
        featureFile.write("\n")
        i += 1
