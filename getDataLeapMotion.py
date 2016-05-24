import sys, thread
sys.path.append("/Users/admin/Desktop/LeapSDK/lib/")
import Leap
from Leap import Finger

'''
Listener
Connects to the Leap Motion and extracts the data.
'''
class Listener(Leap.Listener):
    # Set default positions to -999 in order to know
    # when the Leap Motion has not found hands.
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


'''
start()
Returns in format: [x1,d2,y3],[x4,d5,y6],[[x7,d8,9y],[x1,d2,y3],...[xn,dn,yn]]
Where the first list is the position of the left palm.
The second list is the position of the right palm.
The nested list contains the position of each finger.
'''
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
        # Save the data into lists, from numpy.arrays
        for x in range(3):
            leftPalm.append(listener.left_hand_position[x])
        for x in range(3):
            rightPalm.append(listener.right_hand_position[x])
        for x in range(10):
            fingers.append(listener.finger_positions)

        # Remove the sample listener when done
        controller.remove_listener(listener)

    returnValues.append(leftPalm)
    returnValues.append(rightPalm)
    returnValues.append(fingers)
    return returnValues

'''
getDataLeapMotion connects to the Leap Motion and extracts the data:
position of palms and fingers
'''
if __name__ == "__main__":
    start()
