import math
import json
import getPositions

def calculateDistances(rightPalmPosition, rightFingerTipPositions, leftPalmPosition, leftFingerTipPositions):
    #print len(rightPalmPosition), len(rightFingerTipPositions), len(leftPalmPosition), len(leftFingerTipPositions)
    featureVector = []
    if rightPalmPosition != [] and rightFingerTipPositions != []:
        for n in range(0, len(rightFingerTipPositions)):
            rightCoordinates = rightFingerTipPositions[n]
            # Extract the distance in each direction
            x = abs(rightCoordinates[0] - rightPalmPosition[0])
            y = abs(rightCoordinates[1] - rightPalmPosition[1])
            z = abs(rightCoordinates[2] - rightPalmPosition[2])

            # Add the euclidian distance to the feature vector
            featureVector.append(euclidianDistance(x,y,z))

    if (leftPalmPosition != [] and leftFingerTipPositions != []):
        for n in range(0, len(leftFingerTipPositions)):
            leftCoordinates = leftFingerTipPositions[n]
            # Extract the distance in each direction
            x = abs(leftCoordinates[0] - leftPalmPosition[0])
            y = abs(leftCoordinates[1] - leftPalmPosition[1])
            z = abs(leftCoordinates[2] - leftPalmPosition[2])

            # Add the euclidian distance to the feature vector
            featureVector.append(euclidianDistance(x,y,z))


    return featureVector

def euclidianDistance(x,y,z):
    return math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))


print('---------------------')
print(calculateDistances(*getPositions.getHandPositions('sample1.json')))
print('---------------------')
print(calculateDistances(*getPositions.getHandPositions('sample2.json')))
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
