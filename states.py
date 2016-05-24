import time, sys, os

# Print function to get right indents
def printout(self, toPrint):
	sys.stdout.write("\r {:<50s}".format(toPrint))
	printCart(self.cart)

# Print what is in the cart
def printCart(cart):
	print("{:>50}{}".format('Cart: ', ', '.join(cart)))

'''
Gstate contains the state in the program flow and the cart.
It also contains the main UI progrm flow function, GestureState.
'''
class Gstate:
	current_state = 0
	cart = []
	gestures = {0: 'INIT', 1: 'ALCOHOL', 2: 'NON-ALCOHOL', 3:'FOOD', 4: 'UNDO', 5:'CHECKOUT', 6:'CASH', 7:'CREDIT CARD'}


	'''
	GestureState is the UI and main program flow.
	Input arguments: class state and gesture as int in the range [0, 7]
	'''
	def GestureState(self, gesture):
		if gesture == 0 and self.current_state == 0:
			self.current_state = 1
			os.system("say 'What would you like to order?'")
			printout(self, "What would you like to order?")

		elif gesture == 1 and self.current_state == 1:
			self.cart.append(self.gestures[gesture])
			os.system("say 'Alcoholic drink added!'")
			printout(self, "Alcoholic drink added!")

		elif gesture == 2 and self.current_state == 1:
			self.cart.append(self.gestures[gesture])
			os.system("say 'Non-alcohol drink added!'")
			printout(self, "Non-alcohol drink added!")

		elif gesture == 3 and self.current_state == 1:
			os.system("say 'Food added!'")
			self.cart.append(self.gestures[gesture])
			printout(self, "{} added.".format(self.gestures[gesture]))

		elif gesture == 4 and self.current_state == 1 and len(self.cart) > 0:
			removedItem = self.cart[len(self.cart)-1]
			os.system("say 'Undoing!'")
			self.cart.pop()
			printout(self, "Removed {} from cart".format(removedItem))

		elif gesture == 4 and self.current_state == 1 and len(self.cart) == 0:
			self.current_state = 0
			os.system("say 'Nothing more to remove, going back. Welcome back'")
			printout(self, "Nothing more to remove, going back....... Welcome back")

		elif gesture == 5 and self.current_state == 1:
			self.current_state = 2
			os.system("say 'How would you like to pay?'")
			printout(self, "How would you like to pay?")

		elif gesture == 6 and self.current_state == 2:
			os.system("say 'You paid with cash, cash is king!'")
			printout(self, "You paid for {} and payed with {}".format(", ".join(self.cart), self.gestures[gesture]))
			self.cart = []
			self.current_state = 0
			return 99

		elif gesture == 7 and self.current_state == 2:
			os.system("say 'You paid with card!'")
			printout(self, "You paid for {} and payed with {}".format(", ".join(self.cart), self.gestures[gesture]))
			self.cart = []
			self.current_state = 0
			return 99

		elif gesture == 5 and self.current_state == 2:
			self.current_state = 1
			os.system("say 'Back to order'")
			printout(self, "Back to order!")
