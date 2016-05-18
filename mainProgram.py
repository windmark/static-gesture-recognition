import time, sys
import states
import getDataLeapMotion
from training.training import Knn

# Get input from Leap Motion
def getDataFromLeapMotion():
    sampleData(getDataLeapMotion.start())

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

runUI(1)
