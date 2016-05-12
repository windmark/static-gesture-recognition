import time
import sys

def timeCount():
    sleepTime = 1

    print ("Sleeping for {} seconds".format(sleepTime))
    for sleeping in range(sleepTime, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining.".format(sleeping))
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rComplete!            \n")



if __name__ == "__main__":
    dict = {1: 'init', 2: 'alcohol', 3: 'non-alcohol', 4:'food', 5: 'undo', 6:'checkout', 7:'cash', 8:'credit'}
    if len(sys.argv) > 1:
        leftPalm = [2,5,7]
        strLeftPalm = str(leftPalm)
        print dict[int(sys.argv[1])]
        with open('rawData.txt', 'w') as file_:
            file_.write("ALCOHOL:{}:[1,2,3]:[[x,d,y][x,d,y]...[x,d,y]] \n".format(strLeftPalm))
