import numpy as np
from random import uniform

class NeuralNet(object):

	def __init__(self, numip, numh, numop, lrate=0.3):
		np.random.seed(1)
		self.numip = numip
		self.numh = numh
		self.numop = numop
		self.lrate = lrate
		self.weights_h = None
		self.weights_op = None

	#Sigmoid activation function
	def sigmoid(self, x): 
		return 1/(1 + np.exp(-x))

	#Differential of sigmoid function
	def dsigmoid(self, x):
		return x * (1 - x)

	def predict(self, input):
		temp = self.sigmoid(np.dot(input, self.weights_h))
		return (self.sigmoid(np.dot(temp,self.weights_op)))

	def train(self, inputs, targets, epochs=30000):

		self.X = np.array(inputs)        
		self.Y = np.array(targets)

		#Initialize weights and bias
		self.weights_h1 = 2*np.random.uniform(-1, 1, (self.numip, self.numh)) - 1
		self.weights_op = 2*np.random.uniform(-1, 1, (self.numh, self.numop)) - 1

		for j in range(epochs):

			#Feed forward
		    temp = self.sigmoid(np.dot(self.X, self.weights_h))
		    out = self.sigmoid(np.dot(temp, self.weights_op))

		    #Error of output layer
		    out_error = self.Y - out

		    mean_error = str(np.mean(np.abs(out_error)))

		    if (j%(epochs/10)) == 0:
		        print "Current error : " + mean_error
		        
		    #Delta of output layer
		    out_delta = out_error * self.dsigmoid(out)

		    #Error of hidden layer
		    hidden_error = out_delta.dot(self.weights_op.T)
		    
		    #Delta of hidden layer
		    hidden_delta = hidden_error * self.dsigmoid(temp)

		    self.weights_op += temp.T.dot(out_delta) * self.lrate
		    self.weights_h += self.X.T.dot(hidden_delta) * self.lrate

		hidden_wtext = np.asarray(self.weights_h)
		output_wtext = np.asarray(self.weights_op)
		np.savetxt('hidden_weights.csv', hidden_wtext, delimiter=',')
		np.savetxt('output_weights.csv', output_wtext, delimiter=',')


