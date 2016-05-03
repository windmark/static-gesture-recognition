import math
import json
import getPositions

def calculateDistances(rightPalmPosition, rightFingerTipPositions, leftPalmPosition, leftFingerTipPositions):
	featureVector = []
	for n in range(0, len(rightFingerTipPositions)):
		coordinates = rightFingerTipPositions[n]
		# Extract the distance in each direction
		x = abs(coordinates[0] - rightPalmPosition[0])
		y = abs(coordinates[1] - rightPalmPosition[1])
		z = abs(coordinates[2] - rightPalmPosition[2])

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
