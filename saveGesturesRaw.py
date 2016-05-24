import sys, thread
import getDataLeapMotion


'''
toFile()

'''
def toFile(gestureNr):
    gestures = {0: 'INIT', 1: 'ALCOHOL', 2: 'NON-ALCOHOL', 3:'FOOD', 4: 'UNDO', 5:'CHECKOUT', 6:'CASH', 7:'CREDIT'}
    gesture = gestures[gestureNr]
    print '-------------------------------------'
    print "Attempting gesture {}".format(gesture)

    #Get data from Leap Motion
    leftPalm, rightPalm, fingers = getDataLeapMotion.start()

    # Check if real hands were found.
    if min(leftPalm) == -999 or min(rightPalm) == -999:
        print "Couldn't find two hands!"
        # Restart process if two hands were not found
        userInput = raw_input("Do you want to restart?")

        if userInput == '':
            toFile(gestureNr)
        else:
            print "Bye!"
    else:
        with open('rawData.txt', 'a') as text_file:
            # Save to file (adding to what is already in the file)
            text_file.write("{}\t{}\t{}\t{}\n".format(gesture,leftPalm,rightPalm,fingers))
    return True


'''
saveGesturesRaw
Connects to Leap Motion (through getDataLeapMotion)
Saves the raw data of the gesture to rawData.txt

Input arguments: int in the range [0, 7]
The input arguments tells what gesture to classify as.
'''
if __name__ == "__main__":
    if len(sys.argv) > 1:
        toFile(int(sys.argv[1]))
