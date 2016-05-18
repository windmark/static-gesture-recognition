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
        processedData = dataProcessing.readRawDataAsArguments(rawLeftPalmData, rawRightPalmData, rawFingerData)
        print "got here {}".format(processedData)
        sampleData(processedData)

# Sample input every 1/2 seconds
def sampleData(LMData):
    #TODO How many inputs per seconds do we get?
    sampledData = LMData #TODO sample this first!
    useClassifier(sampledData)

# Throw samples at classifier
def useClassifier(sampledData):
    #Classify should return a gesture int 1 to 8
    runUI(training.Knn())

# Run states.py with gesture
def runUI(gesture):
    gs = states.Gstate()
    gs.GestureState(gesture)

getDataFromLeapMotion()
