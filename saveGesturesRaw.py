import sys, thread, time
#import getDataLeapMotion

def start2():
    return [1,2,3],[4,5,-99],[[7,8,9],[1,2,3]]

def toFile(gestureNr):
    dict = {1: 'INIT', 2: 'ALCOHOL', 3: 'NON-ALCOHOL', 4:'FOOD', 5: 'UNDO', 6:'CHECKOUT', 7:'CASH', 8:'CREDIT'}
    gesture = dict[gestureNr]
    print "Saving gesture {}".format(gesture)

    leftPalm, rightPalm, fingers = start2()
    #leftPalm, rightPalm, fingers = getDataLeapMotion.start()

    if min(leftPalm) == -999 or min(rightPalm) == -999:
        print "Couldn't find two hands!"
        userInput = raw_input("Do you want to restart? Press ONLY enter.\nELSE press a character then enter\n")
        if userInput == '':
            toFile(gestureNr)
        else:
            print "Bye!"
    else:
        with open('rawData.txt', 'a') as text_file:
            text_file.write("{}\t{}\t{}\t{}\n".format(gesture,leftPalm,rightPalm,fingers))
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        toFile(int(sys.argv[1]))
