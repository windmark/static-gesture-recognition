import time, sys

def printout(self, toPrint):
	sys.stdout.write("\r {:<50s}".format(toPrint))
	printCart(self.cart)
	#sys.stdout.flush()

def printCart(cart):
	print("{:>50}{}".format('Cart: ', ', '.join(cart)))
	#sys.stdout.write("{:>75}{}".format('Cart: ', ', '.join(cart)))
	#sys.stdout.flush()

class Gstate:

	current_state = 0
	cart = []
	gestures = {1: 'INIT', 2: 'ALCOHOL', 3: 'NON-ALCOHOL', 4:'FOOD', 5: 'UNDO', 6:'CHECKOUT', 7:'CASH', 8:'CREDIT CARD'}

	def GestureState(self, gesture):
		if gesture == 1 and self.current_state == 0:
			self.current_state = 1
			printout(self, "What would you like to order?")

		elif gesture == 2 and self.current_state == 1:
			self.cart.append(self.gestures[gesture])
			printout(self, "Alcoholic drink added!")

		elif gesture == 3 and self.current_state == 1:
			self.cart.append(self.gestures[gesture])
			printout(self, "Non-alcohol drink added!")

		elif gesture == 4 and self.current_state == 1:
			self.cart.append(self.gestures[gesture])
			printout(self, "{} added.".format(self.gestures[gesture]))

		elif gesture == 5 and self.current_state == 1 and len(self.cart) > 0:
			removedItem = self.cart[len(self.cart)-1]
			self.cart.pop()
			printout(self, "Removed {} from cart".format(removedItem))

		elif gesture == 5 and self.current_state == 1 and len(self.cart) == 0:
			self.current_state = 0
			printout(self, "Nothing more to remove, going back....... Welcome back")

		elif gesture == 6 and self.current_state == 1:
			self.current_state = 2
			printout(self, "How would you like to pay?")

		elif gesture == 7 and self.current_state == 2:
			printout(self, "You paid for {} and payed with {}".format(", ".join(self.cart), self.gestures[gesture]))
			self.cart = []
			self.current_state = 0

		elif gesture == 8 and self.current_state == 2:
			printout(self, "You paid for {} and payed with {}".format(", ".join(self.cart), self.gestures[gesture]))
			self.cart = []
			self.current_state = 0

		elif gesture == 5 and self.current_state == 2:
			self.current_state = 1
			printout(self, "Back to order!")

		'''
		else:
			printout(self, "Gesture {} not legit!".format(self.gestures[gesture]))
			pass
		'''
############################################### TEST ##################################################

def timeToSleep():
	time.sleep(0)

gs = Gstate()

gs.GestureState(0)
timeToSleep()
gs.GestureState(1)
timeToSleep()
gs.GestureState(4)
timeToSleep()
gs.GestureState(2)
timeToSleep()
gs.GestureState(2)
timeToSleep()
gs.GestureState(4)
gs.GestureState(6)
timeToSleep()
gs.GestureState(7)
#print "\n"
