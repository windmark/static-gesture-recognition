import time, sys
import states



# Get input from Leap Motion
def getDataFromLeapMotion():
    sampleData(getDataLeapMotion.start())

# Sample input every 1/2 seconds
def sampleData(LMData):
    pass

# Import classifier
def importClassifer():
    pass

# Throw samples at classifier
def useClassifier():
    pass

# Run states.py with gesture
def runUI(gesture):
    gs = states.Gstate()
    gs.GestureState(gesture)

runUI(1)
