import sys, thread
sys.path.append("/Users/babaktr/LeapSDK/lib/")
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
    leftPalm = []
    rightPalm = []
    fingers = []

    # Keep this process running until Enter is pressed
    
    print "Press Enter to save"
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        #print listener.left_hand_position
        for x in range(3):
            leftPalm.append(listener.left_hand_position[x])
        for x in range(3):
            rightPalm.append(listener.right_hand_position[x])
        for x in range(10):
            fingers.append(listener.finger_positions)
        controller.remove_listener(listener)
        #print returnValues
    #Return in format: [1,2,3],[4,5,6],[[7,8,9],[1,2,3],...[x,d,y]]
    returnValues.append(leftPalm)
    returnValues.append(rightPalm)
    returnValues.append(fingers)
    return returnValues

if __name__ == "__main__":
    start()
