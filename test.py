import math
import json
from pprint import pprint
 
def calculateDistances(palmPosition, fingerTipPositions):
	featureVector = []
	for n in range(0, len(fingerTipPositions)):
		coordinates = fingerTipPositions[n]
		# Extract the distance in each direction
		x = abs(coordinates[0] - palmPosition[0])
		y = abs(coordinates[1] - palmPosition[1])
		z = abs(coordinates[2] - palmPosition[2])
		
		# Add the euclidian distance to the feature vector
		featureVector.append(euclidianDistance(x,y,z))

	return featureVector

def euclidianDistance(x,y,z):
	return math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))

def getPositions(filename):
    #input data
    with open(filename) as data_file:
        data = json.load(data_file)
 
    #hand is "right" or "left"
    rightLeft = data["hands"][0]["type"]
 
    #position of palm center in format [x,distance,y] (or y,distance,x?)
    palmPosition = data["hands"][0]["palmPosition"]
 
    #array with all fingertips in format [x,distance,y]
    dPhalanges = []
    for x in range(5):
        dPhalanges.append(data["pointables"][x]["btipPosition"])
 
 
    featureVector = calculateDistances(palmPosition,dPhalanges)
    print(featureVector)
 
print('---------------------')
getPositions('sample1.json')
print('---------------------')
getPositions('sample2.json')
print('---------------------')
getPositions('sample3.json')
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