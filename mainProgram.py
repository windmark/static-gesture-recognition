import time, sys
import states
import getDataLeapMotion
import dataProcessing
from training.training import Knn

# Get input from Leap Motion
def getDataFromLeapMotion():
    rawLeftPalmData,rawRightPalmData,rawFingerData = getDataLeapMotion.start()
    if min(rawLeftPalmData) == -999 or min(rawRightPalmData) == -999:
        print "Couldn't find two hands!"
        getDataFromLeapMotion()
    else:
        #print "got here"
        #print rawFingerData[0][:10]
        rawFingerData2 = rawFingerData[0][:30]
        processedData = dataProcessing.convertToFeatureVectors([rawLeftPalmData],[rawRightPalmData],[rawFingerData2])
        #print "processedData: {}".format(processedData[0])
        useClassifier(processedData[0])

# Sample input every 1/2 seconds
def sampleData(LMData):
    #TODO How many inputs per seconds do we get?
    sampledData = LMData #TODO sample this first!
    useClassifier(sampledData)

# Throw samples at classifier
def useClassifier(sampledData):
    #Classify should return a gesture int 1 to 8

    gesture = knn.classify(sampledData)
    print(gesture)
    runUI(gesture)

# Run states.py with gesture
def runUI(gesture):
    end = gs.GestureState(gesture)
    if end != 99:
        getDataFromLeapMotion()
    else:
        return True

gs = states.Gstate()
knn = Knn()
knn.train('training/features.txt')
getDataFromLeapMotion()
