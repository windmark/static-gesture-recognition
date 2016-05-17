import time, sys

def printout(self, toPrint):
	#print "\r {:<30s}         ".format(toPrint)
	sys.stdout.write("\r {:<35s}".format(toPrint))
	printCart(self.cart)
	#sys.stdout.flush()

def printCart(cart):
	print "{:>25}:{}".format("Cart:",cart)
	#sys.stdout.write("{:>25}:{}".format("Cart:",cart))
	#sys.stdout.flush()

class Gstate:

	current_state = 0
	cart = []
	dict = {1: 'INIT', 2: 'ALCOHOL', 3: 'NON-ALCOHOL', 4:'FOOD', 5: 'UNDO', 6:'CHECKOUT', 7:'CASH', 8:'CREDIT'}


	def GestureState(self, gesture):
		if gesture == "init" and self.current_state == 0:
			self.current_state = 1
			printout(self, "What would you like to order?")

		elif gesture == "alcohol" and self.current_state == 1:
			self.cart.append("alcoholic-drink")
			printout(self, "Alkohol drink added!")

		elif gesture == "non-alcohol" and self.current_state == 1:
			self.cart.append("non-alcoholic-drink")
			printout(self, "Non-alkohol drink added!")

		elif gesture == "food" and self.current_state == 1:
			self.cart.append("food")
			printout(self, "food added!")

		elif gesture == "undo" and self.current_state == 1 and len(self.cart) > 0:
			self.cart.pop()
			printout(self, "removed last item.")

		elif gesture == "undo" and self.current_state == 1 and len(self.cart) == 0:
			self.current_state = 0
			printout(self, "Nothing more to remove, going back....... Welcome back")

		elif gesture == "checkout" and self.current_state == 1:
			self.current_state = 2
			printout(self, "How would you like to pay?")

		elif gesture == "cash" and self.current_state == 2:
			self.cart = []
			self.current_state = 0
			printout(self, "You paid for: " + str(self.cart)[1:-1] + "and payed with cash!")

		elif gesture == "credit" and self.current_state == 2:
			self.cart = []
			self.current_state = 0
			printout(self, "You paid for: " + str(self.cart) + "and payed with card!")

		elif gesture == "undo" and self.current_state == 2:
			self.current_state = 1
			printout(self, "Back to order!")

############################################### TEST ##################################################

gs = Gstate()
gs.GestureState("init")
time.sleep(1)
gs.GestureState("alcohol")
time.sleep(1)
gs.GestureState("non-alcohol")
time.sleep(1)
gs.GestureState("checkout")
time.sleep(1)
gs.GestureState("cash")

gs.GestureState("init")
time.sleep(1)
gs.GestureState("alcohol")
time.sleep(1)
gs.GestureState("non-alcohol")
time.sleep(1)
gs.GestureState("checkout")
time.sleep(1)
gs.GestureState("cash")
print "\n"
