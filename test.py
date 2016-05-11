import math
import json
import getPositions

def calculateDistances(rightPalmPosition, rightFingerTipPositions, leftPalmPosition, leftFingerTipPositions):
    #print len(rightPalmPosition), len(rightFingerTipPositions), len(leftPalmPosition), len(leftFingerTipPositions)
    featureVector = []
    if rightPalmPosition != [] and rightFingerTipPositions != []:
        for n in range(0, len(rightFingerTipPositions)):
            #rightFingerPosition = rightFingerTipPositions[n]
            # Extract the distance in each direction
            x = abs(rightFingerTipPositions[n][0] - rightPalmPosition[0])
            #d = abs(rightFingerTipPositions[n][1] - rightPalmPosition[1])
            y = abs(rightFingerTipPositions[n][2] - rightPalmPosition[2])

            # Add the euclidian distance to the feature vector
            fingerDistance = euclidianDistance(x, y)
            #print fingerDistance

            featureVector.append(euclidianDistance(x,y))


    if (leftPalmPosition != [] and leftFingerTipPositions != []):
        for n in range(0, len(leftFingerTipPositions)):
            #leftFingerPosition = leftFingerTipPositions[n]
            # Extract the distance in each direction
            x = abs(leftFingerTipPositions[n][0] - leftPalmPosition[0])
            #d = abs(leftFingerTipPositions[n][1] - leftPalmPosition[1])
            y = abs(leftFingerTipPositions[n][2] - leftPalmPosition[2])

            # Add the euclidian distance to the feature vector
            featureVector.append(euclidianDistance(x,y))

    return featureVector

def euclidian3DDistance(x,y,z):
    return math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))

def euclidianDistance(x,y):
    return math.sqrt(math.pow(x,2) + math.pow(y,2))

def normalize(array):
    normalizedData = []
    for n in range(0,2):
        try:
            for a in range(5):
                tmp = array[a]
                minval = min(array[n*5:(n+1)*5])
                maxval= max(array[n*5:(n+1)*5])
                #normalizedData.append((tmp - minval) / (maxval - minval))
                #rounding to 5 decimals
                normalizedData.append(round((tmp - minval) / (maxval - minval),5))
        except:
            pass
    return normalizedData


print('---------------------')
print normalize(calculateDistances(*getPositions.getHandPositions('sample1.json')))
print('---------------------')
print normalize(calculateDistances(*getPositions.getHandPositions('sample2.json')))
print('---------------------')
print(calculateDistances(*getPositions.getHandPositions('sample3.json')))
print('---------------------')



# palmPosition = [0,0,0]
# openPalm = [[100,100,100], [100,100,100], [100,100,100], [100,100,100], [100,100,100]]
# closedPalm = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]

# print('----------')
# featureVector = calculateDistances(palmPosition, openPalm)

# print(featureVector)

# print('----------')

# featureVector = calculateDistances(palmPosition, closedPalm)

# print(featureVector)
