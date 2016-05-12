import time
import sys

sleepTime = 5

print ("Sleeping for {} seconds".format(sleepTime))
for sleeping in range(sleepTime, 0, -1):
    sys.stdout.write("\r")
    sys.stdout.write("{:2d} seconds remaining.".format(sleeping))
    sys.stdout.flush()
    time.sleep(1)
sys.stdout.write("\rComplete!            \n")
