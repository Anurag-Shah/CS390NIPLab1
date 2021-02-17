
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import pandas as pd


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # Replacement for above line

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"

class NeuralNetwork_2Layer():
	def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1, activation = "sigmoid"):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.neuronsPerLayer = neuronsPerLayer
		self.lr = learningRate
		self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
		self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

		self.activation = activation

	# Activation functions

	def __activation(self, x):
		if self.activation == "sigmoid":
			return self.__sigmoid(x)
		elif self.activation == "relu":
			return self.__relu(x)
		else:
			raise Exception("Invalid Activation Function")

	def __activationDerivative(self, x):
		if self.activation == "sigmoid":
			return self.__sigmoidDerivative(x)
		elif self.activation == "relu":
			return self.__reluDerivative(x)
		else:
			raise Exception("Invalid Activation Function")

	def __relu(self, x):
		return np.maximum(x, 0)

	def __sigmoid(self, x):
		# Without the clip, we get an overflow error.
		# np.exp can handle a max of approximately 708 from my testing, so cutting it a bit short
		x2 = np.clip(x, -700, 700)
		return (1 / (1 + np.exp(-1 * x2)))

	# Activation prime functions
	def __reluDerivative(self, x):
		x[x <= 0] = 0
		x[x > 0] = 1
		return x

	def __sigmoidDerivative(self, x):
		return x * (1 - x)

	# Batch generator for mini-batches. Not randomized.
	def __batchGenerator(self, l, n):
		for i in range(0, len(l), n):
			yield l[i : i + n]

	# Loss function (half mse)
	def __mseloss(self, actual, predicted):
		squares = np.square(actual - predicted)
		fullmse = np.sum(squares) / len(squares)
		return 0.5 * fullmse

	# Training with backpropagation.
	def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
		if minibatches == True:
			self.mbTrain(xVals, yVals, epochs, mbs)
		else:
			self.fullTrain(xVals, yVals, epochs)

	# Training on minibatches
	def mbTrain(self, xVals, yVals, epochs, mbs):
		for i in range(0, epochs):
			inputbatches = self.__batchGenerator(xVals, mbs)
			labelbatches = self.__batchGenerator(yVals, mbs)
			batches = zip(inputbatches, labelbatches)
			cumloss = float(0)
			for inputbatch, labelbatch in batches:
				outputs = self.__forward(inputbatch)
				cumloss += self.__mseloss(labelbatch, outputs[1])
				l2e = outputs[1] - labelbatch
				l2d = l2e * self.__activationDerivative(outputs[1])
				l1e = l2d @ self.W2.T
				l1d = l1e * self.__activationDerivative(outputs[0])
				self.W1 -= (inputbatch.T @ l1d) * self.lr
				self.W2 -= (outputs[0].T @ l2d) * self.lr
			inputlen = xVals.shape[0]
			finalloss = cumloss * mbs / inputlen

	# Training on full data set
	def fullTrain(self, xVals, yVals, epochs):
		for i in range(epochs):
			outputs = self.__forward(xVals)
			# index [0] is layer 1 output, index [1] is layer 2 output
			finalLoss = self.__mseloss(yVals, outputs[1])
			l2e = outputs[1] - yVals
			l2d = l2e * self.__activationDerivative(outputs[1])
			l1e = l2d @ self.W2.T
			l1d = l1e * self.__activationDerivative(outputs[0])
			self.W2 -= (outputs[0].T @ l2d) * self.lr
			self.W1 -= (xVals.T @ l1d) * self.lr

	# Forward pass.
	def __forward(self, input):
		layer1 = self.__activation(np.dot(input, self.W1))
		layer2 = self.__activation(np.dot(layer1, self.W2))
		return layer1, layer2

	# Predict.
	def predict(self, xVals):
		_, layer2 = self.__forward(xVals)
		onehotOutputs = np.zeros_like(layer2)
		onehotOutputs[np.arange(len(layer2)), layer2.argmax(1)] = 1
		return onehotOutputs

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
	ans = []
	for entry in xTest:
		pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		pred[random.randint(0, 9)] = 1
		ans.append(pred)
	return np.array(ans)

#=========================<Pipeline Functions>==================================

def getRawData(dataset = "mnist"):
	if dataset == "mnist":
		mnist = tf.keras.datasets.mnist
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
		print("Shape of xTrain dataset: %s." % str(xTrain.shape))
		print("Shape of yTrain dataset: %s." % str(yTrain.shape))
		print("Shape of xTest dataset: %s." % str(xTest.shape))
		print("Shape of yTest dataset: %s." % str(yTest.shape))
		return ((xTrain, yTrain), (xTest, yTest))
	elif dataset == "iris":
		return pd.read_csv("irisdata.csv", header=None)
	else:
		raise Exception("Invalid Dataset")

def preprocessData(raw, dataset = "mnist"):
	if dataset == "mnist":
		((xTrain, yTrain), (xTest, yTest)) = raw
		xTrain = np.reshape(xTrain, [-1, IMAGE_SIZE])
		xTrain = xTrain / 255.0
		xTest = np.reshape(xTest, [-1, IMAGE_SIZE])
		xTest = xTest / 255.0
		yTrainP = to_categorical(yTrain, NUM_CLASSES)
		yTestP = to_categorical(yTest, NUM_CLASSES)
		print("New shape of xTrain dataset: %s." % str(xTrain.shape))
		print("New shape of xTest dataset: %s." % str(xTest.shape))
		print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
		print("New shape of yTest dataset: %s." % str(yTestP.shape))
		return ((xTrain, yTrainP), (xTest, yTestP))

	elif dataset == "iris":

		# Randomize and split into train and test

		raw = raw.sample(frac = 1)
		trainRaw = raw.sample(frac = 0.8)
		testRaw = raw.drop(trainRaw.index)

		# Split X and Label
		xTrain = trainRaw[trainRaw.columns[0:4]]
		yTrainRaw = trainRaw[trainRaw.columns[4]]
		xTest = testRaw[testRaw.columns[0:4]]
		yTestRaw = testRaw[testRaw.columns[4]]

		xTrain = xTrain.to_numpy()
		xTest = xTest.to_numpy()
		yTrainRaw = yTrainRaw.to_numpy()
		yTestRaw = yTestRaw.to_numpy()

		# Preprocess X (scale to 0-1)
		maxCol0 = np.max(raw[raw.columns[0]])
		maxCol1 = np.max(raw[raw.columns[1]])
		maxCol2 = np.max(raw[raw.columns[2]])
		maxCol3 = np.max(raw[raw.columns[3]])

		xTrain[:,0] /= maxCol0
		xTrain[:,1] /= maxCol1
		xTrain[:,2] /= maxCol2
		xTrain[:,3] /= maxCol3
		xTest[:,0] /= maxCol0
		xTest[:,1] /= maxCol1
		xTest[:,2] /= maxCol2
		xTest[:,3] /= maxCol3

		yTrain = []
		yTest = []

		sertosa = np.array([1, 0, 0])
		versicolor = np.array([0, 1, 0])
		virginica = np.array([0, 0, 1])

		# Y Train
		for label in yTrainRaw:
			if label == "Iris-setosa":
				yTrain.append(sertosa)
			elif label == "Iris-versicolor":
				yTrain.append(versicolor)
			elif label == "Iris-virginica":
				yTrain.append(virginica)

		# Y Test
		for label in yTestRaw:
			if label == "Iris-setosa":
				yTest.append(sertosa)
			elif label == "Iris-versicolor":
				yTest.append(versicolor)
			elif label == "Iris-virginica":
				yTest.append(virginica)

		yTrain = np.array(yTrain)
		yTest = np.array(yTest)

		print("Shape of xTrain dataset (Iris): %s." % str(xTrain.shape))
		print("Shape of xTest dataset (Iris): %s." % str(xTest.shape))
		print("Shape of yTrain dataset (Iris): %s." % str(yTrain.shape))
		print("Shape of yTest dataset (Iris): %s." % str(yTest.shape))

		return ((xTrain, yTrain), (xTest, yTest))

	else:
		raise Exception("Invalid Dataset")

def trainModel(data):
	xTrain, yTrain = data
	if ALGORITHM == "guesser":
		return None   # Guesser has no model, as it is just guessing.
	elif ALGORITHM == "custom_net":
		insize = xTrain.shape
		outsize = yTrain.shape
		net = NeuralNetwork_2Layer(insize[1], outsize[1], neuronsPerLayer = 30, activation = "sigmoid")
		net.train(xTrain, yTrain, epochs = 15, minibatches = True, mbs = 50)
		return net
	elif ALGORITHM == "tf_net":
		categorical_crossentropy = keras.losses.CategoricalCrossentropy()
		opt = keras.optimizers.Adam(learning_rate = 0.01)
		net = keras.models.Sequential()
		net.add(tf.keras.layers.Dense(140, activation = tf.nn.sigmoid))
		net.add(keras.layers.Dense(10, activation = tf.nn.sigmoid))
		net.compile(optimizer = opt, loss = categorical_crossentropy, metrics = [keras.metrics.CategoricalAccuracy()])
		net.fit(xTrain, yTrain, epochs = 30, verbose = 0)
		return net
	else:
		raise ValueError("Algorithm not recognized.")

def trainIrisModel(data):
	xTrain, yTrain = data
	if ALGORITHM == "guesser":
		return None   # Guesser has no model, as it is just guessing.
	elif ALGORITHM == "custom_net":
		insize = xTrain.shape
		outsize = yTrain.shape
		net = NeuralNetwork_2Layer(insize[1], outsize[1], neuronsPerLayer = 30, activation = "sigmoid")
		net.train(xTrain, yTrain, epochs = 40, minibatches = True, mbs = 3)
		return net
	elif ALGORITHM == "tf_net":
		categorical_crossentropy = keras.losses.CategoricalCrossentropy()
		opt = keras.optimizers.Adam(learning_rate = 0.01)
		net = keras.models.Sequential()
		net.add(tf.keras.layers.Dense(40, activation = tf.nn.sigmoid))
		net.add(keras.layers.Dense(3, activation = tf.nn.sigmoid))
		net.compile(optimizer = opt, loss = categorical_crossentropy, metrics = [keras.metrics.CategoricalAccuracy()])
		net.fit(xTrain, yTrain, epochs = 100, verbose = 0)
		return net
	else:
		raise ValueError("Algorithm not recognized.")

def runModel(data, model):
	if ALGORITHM == "guesser":
		return guesserClassifier(data)
	elif ALGORITHM == "custom_net":
		print("Testing Custom_NN.")
		return model.predict(data)
	elif ALGORITHM == "tf_net":
		layer2 = model.predict(data)
		onehotOutputs = np.zeros_like(layer2)
		onehotOutputs[np.arange(len(layer2)), layer2.argmax(1)] = 1
		return onehotOutputs
	else:
		raise ValueError("Algorithm not recognized.")

def evalResults(data, preds):
	xTest, yTest = data
	acc = 0

	for i in range(preds.shape[0]):
		if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
	accuracy = acc / preds.shape[0]
	print("Classifier algorithm: %s" % ALGORITHM)
	print("Classifier accuracy: %f%%" % (accuracy * 100))
	print()

	# Initialize confusion matrix
	matrixSize = preds.shape[1]
	confusionMatrix = np.zeros([matrixSize, matrixSize])

	# Populate Matrix
	for i in range(preds.shape[0]):
		# Get the real output, as index in the confusion matrix
		labelIndex = np.argmax(yTest[i])

		# Get the predicted output, as index in the confusion matrix
		predictedIndex = np.argmax(preds[i])

		# Value is at index [label][predicted], because label along X and predicted along Y
		# We increment this value in the matrix
		confusionMatrix[labelIndex][predictedIndex] += 1

	print("Confusion Matrix:")
	print(confusionMatrix.astype(int))

#=========================<Main>================================================

def main():
	print()
	print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
	print("+ Running Model for MNIST +")
	print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
	print()

	raw = getRawData()
	data = preprocessData(raw)
	model = trainModel(data[0])
	preds = runModel(data[1][0], model)
	evalResults(data[1], preds)

	print("\n\n")
	print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
	print("+ Running Model for Iris  +")
	print("=+=+=+=+=+=+=+=+=+=+=+=+=+=")
	print()
	raw = getRawData(dataset = "iris")
	data = preprocessData(raw, dataset = "iris")
	model = trainIrisModel(data[0])
	preds = runModel(data[1][0], model)
	evalResults(data[1], preds)

if __name__ == '__main__':
	main()