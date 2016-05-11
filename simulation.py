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



class SampleListener(Leap.Listener):
    left_hand_position = [-1, -1, -1]
    right_hand_position = [-1, -1, -1]
    finger_positions = [-1]*10

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


def main():
    # Create a sample listener and controller
    listener = SampleListener()
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
        returnValues.append(listener.left_hand_position)
        returnValues.append(listener.right_hand_position)
        returnValues.append(listener.finger_positions)
        controller.remove_listener(listener)

    return returnValues
        


if __name__ == "__main__":
    main()
