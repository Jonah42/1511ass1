# Single-Digit Image Recognition Implemented in C

This project aims to explore exactly how a NN works by constructing a MLP from scratch in C.
Currently 80-85% accurate.

## Input Data
.pbm files, consistently 50x70 pixels. A pool of 1000 test images is divided 80-20 into train and test data respectively.
Of the form ```<number>_<variant>.pbm``` where number is the digit, element [0,9] and variant is element [00,99], note the double digits used.

## MLP
The MLP is a 3 layer network (2 hidden layers). It uses an ReLU activation function and a softmax activation function for the output layer. Derivatives taken as df/dx = (f(x+H)-f(x-H))/2H where H is defined below to be 1e-5.

## Usage
From root directory:

```bin/NN <nodesLayer1> <nodesLayer2> <learningRate> <rateAdjustment> <batchSize> <inputFile>```

Recommended initial values:

	nodesLayer1 = 300
	nodesLayer2 = 100
	learningRate = 0.01
	rateAdjustment = 0.95
	batchSize = 30
	inputFile = in.txt

If no input file is supplied you can run manually, commands are:

```		train data/train/<number>_<variant>```
			- where number element of [0,9] and variant element of [00,79] (note 2 digits)

```		run <path_to_file>```

```		validate```
			- runs all test data against current trained model, and prints probabilities of each digit and guesses which it is

NOTE: I have not put much time into checking cmd line args, if it crashes its on you :P

The program prints the probability of each digit, as well as a guess which is just a max of the probabilities.

## Visualising the output
Piping the output to plot.py will plot the output (or rather save a plot of it). This is designed for the ```validate``` command only. Note there is a variable printed on line 64 of NN.c - this is designed to be the variable you vary when experimenting. Change it based on what you're varying. This variable is used to determine the name of the saved graph.
To vary a parameter and obtain graphs of the results, have a look at the shell script files, notably nodes.sh (the others need to be updated to reflect extra args)

## Other files
gen.py is used to generate in.txt
Makefile just compiles, run ```make```