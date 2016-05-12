################################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################


import sys, thread, time
sys.path.append("/Users/admin/Desktop/LeapSDK/lib/")
import Leap
from Leap import Finger



class Listener(Leap.Listener):
    left_hand_position = [-999, -999, -999]
    right_hand_position = [-999, -999, -999]
    finger_positions = [-999]*10

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        for hand in frame.hands:
            if(hand.is_left):
                self.left_hand_position = hand.palm_position
            if(hand.is_right):
                self.right_hand_position = hand.palm_position

        for pointable in frame.pointables:
            if(pointable.is_finger):
                if(pointable.hand.is_left):
                    hand = 0
                else:
                    hand = 1

                finger = Finger(pointable)
                self.finger_positions[5*hand+finger.type] = [pointable.tip_position[0], pointable.tip_position[1], pointable.tip_position[2]]

                #print("%s %f, %f, %f" % (hand, pointable.tip_position[0], pointable.tip_position[1], pointable.tip_position[2]))


def start():
    # Create a sample listener and controller
    listener = Listener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    returnValues = []

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        #print listener.left_hand_position
        for x in range(3):
            returnValues.append(listener.left_hand_position[x])
        for x in range(3):
            returnValues.append(listener.right_hand_position[x])
        for x in range(10):
            returnValues.append(listener.finger_positions)
        controller.remove_listener(listener)
        #print returnValues

    return returnValues

def convertString(handData):
    #           leftPalm,rightPalm,Fingers
    #handData = [[x,d,y],[x,d,y],[[x,d,y],...*10]]
    handData()
    return ('%a  %a  %a %a' % leftPalmArray


if __name__ == "__main__":
    dict = {1: 'INIT', 2: 'ALCOHOL', 3: 'NON-ALCOHOL', 4:'FOOD', 5: 'UNDO', 6:'CHECKOUT', 7:'CASH', 8:'CREDIT'}
    if len(sys.argv) > 1:
        print dict[int(sys.argv[1])]
        returnValues = start()
        if min(returnValues) == -999:
            print "Hittade inte två händer, sparar inte."
        else:

            with open('rawData.txt', 'w') as file_:
                file_.write(convertString())
