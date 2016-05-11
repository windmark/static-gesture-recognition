import json
import sys

def getHandPositions(filename):
    #input data
    with open(filename) as data_file:
        data = json.load(data_file)

    #Variables
    hand0 = ""
    hand1 = ""
    rightPalmPosition = []
    leftPalmPosition = []
    rightDistalPhalanges = []
    leftDistalPhalanges = []


    try:
        hand0 = data["hands"][0]["type"]

        if hand0 == "right":
            #position of palm center in format [x,distance,y] (or y,distance,x?)
            rightPalmPosition = data["hands"][0]["palmPosition"]
            #array with all fingertips in format [x,distance,y]
            rightDistalPhalanges = []
            for x in range(5):
                rightDistalPhalanges.append(data["pointables"][x]["btipPosition"])
        if hand0 == "left":
            #position of palm center in format [x,distance,y] (or y,distance,x?)
            leftPalmPosition = data["hands"][0]["palmPosition"]
            #array with all fingertips in format [x,distance,y]
            leftDistalPhalanges = []
            for x in range(5):
                leftDistalPhalanges.append(data["pointables"][x]["btipPosition"])

    except:
        #print ("No first hand!")
        #print sys.exc_info()
        pass
    try:
        hand1 = data["hands"][1]["type"]
        if hand1 == "right":
            #position of palm center in format [x,distance,y] (or y,distance,x?)
            rightPalmPosition = data["hands"][1]["palmPosition"]

            #array with all fingertips in format [x,distance,y]
            rightDistalPhalanges = []
            for x in range(5,10):
                rightDistalPhalanges.append(data["pointables"][x]["btipPosition"])
        if hand1 == "left":
            #position of palm center in format [x,distance,y] (or y,distance,x?)
            leftPalmPosition = data["hands"][1]["palmPosition"]

            #array with all fingertips in format [x,distance,y]
            leftDistalPhalanges = []
            for x in range(5,10):
                leftDistalPhalanges.append(data["pointables"][x]["btipPosition"])

    except:
        pass
        #print ("No second hand!")
        #print sys.exc_info()

    return(rightPalmPosition,rightDistalPhalanges,leftPalmPosition,leftDistalPhalanges)
