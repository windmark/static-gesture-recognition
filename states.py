class Gstate: 

	current_state = 0
	cart = []


	def GestureState(self, gesture):
		if gesture == "init" and self.current_state == 0:
			self.current_state = 1
			print "What would you like to order?"
		
		elif gesture == "alcohol" and self.current_state == 1:
			print "Alkohol drink added!"
			self.cart.append("alcoholic-drink")

		elif gesture == "non-alcohol" and self.current_state == 1:
			print "Non-alkohol drink added!"
			self.cart.append("non-alcoholic-drink")

		elif gesture == "food" and self.current_state == 1:
			print "food added!"
			self.cart.append("food")

		elif gesture == "undo" and self.current_state == 1 and len(self.cart) > 0:
			print "removed last item."
			self.cart.pop()

		elif gesture == "undo" and self.current_state == 1 and len(self.cart) == 0:
			print "Nothing more to remove, going back....... Welcome back"
			self.current_state = 0

		elif gesture == "checkout" and self.current_state == 1:
			print "How would you like to pay?"
			self.current_state = 2

		elif gesture == "cash" and self.current_state == 2:
			print "You paid for: " + str(self.cart) + "and payed with cash!"
			self.cart = []
			self.current_state = 0

		elif gesture == "credit" and self.current_state == 2:
			print "You paid for: " + str(self.cart) + "and payed with card!"
			self.cart = []
			self.current_state = 0

		elif gesture == "undo" and self.current_state == 2:
			print "Back to order!"
			self.current_state = 1
			
############################################### TEST ##################################################

gs = Gstate()
gs.GestureState("init")
gs.GestureState("alcohol")
gs.GestureState("checkout")
gs.GestureState("cash")







