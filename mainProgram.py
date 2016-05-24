import time, sys
import states
import getDataLeapMotion
import dataProcessing
from training.training import Knn

# Get input from Leap Motion
def getDataFromLeapMotion():
    # Get data from Leap Motion
    rawLeftPalmData,rawRightPalmData,rawFingerData = getDataLeapMotion.start()

    # Check if the Leap Motion found two hands
    if min(rawLeftPalmData) == -999 or min(rawRightPalmData) == -999:
        print "Couldn't find two hands!"
        # If not, re run function
        getDataFromLeapMotion()
    else:
        # Remove redundant (duplicates) finger data
        rawFingerData2 = rawFingerData[0][:30]
        # Orders data to featureVector format: right palm, right fingers, left palm, left fingers
        # convertToFeatureVectors also normalizes data.
        processedData = dataProcessing.convertToFeatureVectors([rawLeftPalmData],[rawRightPalmData],[rawFingerData2])

        useClassifier(processedData[0])

# Throw samples at classifier
def useClassifier(sampledData):
    #Classify should return a gesture as int in the range [0 to 7]
    gesture = knn.classify(sampledData)
    print(gesture)
    runUI(gesture)

# Run states.py, the UI, with gesture
def runUI(gesture):
    end = gs.GestureState(gesture)
    if end != 99:
        getDataFromLeapMotion()
    else:
        return True


'''

'''
if __name__ == "__main__":
    gs = states.Gstate()
    knn = Knn()
    knn.loadModel('training/knnModel.pkl')
    getDataFromLeapMotion()
