//Three layer MLP designed to tackle numerical single-digit capture recognition
//Written by Jonah Meggs December 2020

//USAGE: from root directory, bin/NN <nodesLayer1> <nodesLayer2> <learningRate> <rateAdjustment> <batchSize> <inputFile>
// If no input file is supplied you can run manually, commands are:
//		train data/train/<number>_<variant>
//			- where number element of [0,9] and variant element of [00,79] (note 2 digits)
//		run <path_to_file>
//		validate
//			- runs all test data against current trained model, and prints probabilities of each digit and guesses which it is

//NOTE: I have not put much time into checking cmd line args, if it crashes its on you :P

//Implemented with reLU activation functions and a softmax activation function for the output layer. Derivatives taken as df/dx = (f(x+H)-f(x-H))/2H
// where H is defined below to be 1e-5

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define WIDTH 50
#define HEIGHT 70
#define SUBWIDTH 25
#define SUBHEIGHT 35
#define INPUT 875
// #define HIDDEN1 300
// #define HIDDEN2 100
#define BIAS 3
#define OUTPUT 10
#define H 0.00001

int HIDDEN1;
int HIDDEN2;

int ** loadInput(char * inFile);
double* runNN(double** w1, double** w2, double** out, double** bias, int ** data);
void trainNN(double** w1, double** w2, double** out, double** bias, int ** data, char expected, double learningRate);
void printWeights(double** w1, double** w2, double** out, int count, double **bias);

int main(int argc, char * argv[]) {
	//Create a 2 layer NN with weights for each layer in arrays. There will be HIDDEN1 nodes in the first layer, HIDDEN2 in the second and 10 output nodes.
	//We have 50*70=3500 inputs. We will sample this to 25*35=875 inputs.
	FILE* fp = stdin;
	if (argc <= 5) {
		printf("Usage: bin/NN <nodesLayer1> <nodesLayer2> <learningRate> <rateAdjustment> <batchSize> <inputFile>");
		exit(0);
	}
	if (argc <= 6) printf("Buenos dias!! Please train or test:");
	else {
		fp = fopen(argv[6], "r");
		if (fp == NULL) {
			printf("Autopilot failed, reverting to manual control\n");
			fp = stdin;
		}
	}
	//Load in all parameters from cmd line args
	HIDDEN1 = atoi(argv[1]);
	HIDDEN2 = atoi(argv[2]);
	double learningRate = atof(argv[3]);
	double rateAdjustment = atof(argv[4]);
	int batch = atoi(argv[5]);
	//Print for graph generation what you want
	printf("%lf\n", learningRate);
	//Allocate weight matrices
	double** w1 = malloc(HIDDEN1*sizeof(double*));
	double** w2 = malloc(HIDDEN2*sizeof(double*));
	double** out = malloc(OUTPUT*sizeof(double*));
	double** bias = malloc(BIAS*sizeof(double*));
	if (w1 == NULL || w2 == NULL || out == NULL || bias == NULL) {
		printf("Weight initialisation failed, exiting\n");
		exit(0);
	}
	//Initialise the weights to a small random number
	double scale = 500.0*sqrt(HIDDEN1);
	for (int i = 0; i < HIDDEN1; i++) {
		w1[i] = malloc(INPUT*sizeof(double));
		for (int j = 0; j < INPUT; j++)
			w1[i][j] = ((rand()%1000)-500)/(scale);
	}
	scale = 500.0*sqrt(HIDDEN2);
	for (int i = 0; i < HIDDEN2; i++) {
		w2[i] = malloc(HIDDEN1*sizeof(double));
		for (int j = 0; j <  HIDDEN1; j++)
			w2[i][j] = ((rand()%1000)-500)/(scale);
	}
	scale = 500.0*sqrt(OUTPUT);
	for (int i = 0; i < OUTPUT; i++) {
		out[i] = malloc(HIDDEN2*sizeof(double));
		for (int j = 0; j < HIDDEN2; j++)
			out[i][j] = ((rand()%1000)-500)/(scale);
	}
	// Bias initialised to 0
	bias[0] = calloc(HIDDEN1, sizeof(double));
	bias[1] = calloc(HIDDEN2, sizeof(double));
	bias[2] = calloc(OUTPUT, sizeof(double));
	//Now train/test at discretion of a script
	//Usage: <action> <inputFile>
	char action[100], inFile[100];
	int expected;
	int **data;
	int trainCount = 0;
	while (1) {
		fscanf(fp, "%s", action);
		if (strcmp(action, "quit") == 0) break;
		else if (strcmp(action, "train") == 0) {
			fscanf(fp, "%s", inFile);
			// printf("Training on %s\n", inFile);
			// Read data in from file
			data = loadInput(inFile);
			// for (int i = 0; i < SUBHEIGHT; i++) {
			// 	for (int j = 0; j < SUBWIDTH; j++)
			// 		printf("%d ", data[i][j]);
			// 	printf("\n");
			// }
			trainNN(w1, w2, out, bias, data, inFile[11]-'0', learningRate);
			//printWeights(w1, w2, out, trainCount);
			for (int i = 0; i < SUBHEIGHT; i++)
				free(data[i]);
			free(data);
			trainCount++;
			//Update learningRate every so often
			if (trainCount%batch == 0) {
				learningRate *= rateAdjustment;
				// trainCount = 0;
			}
			// printf("Finished training\n");
		} else if (strcmp(action, "run") == 0) {
			fscanf(fp, "%s", inFile);
			printf("Running on %s\n", inFile);
			data = loadInput(inFile);
			double* ret = runNN(w1, w2, out, bias, data);
			// Find max
			int guess = 0;
			for (int i = 0; i < OUTPUT; i++) {
				printf("%d: %lf\n", i, ret[i]);
				if (ret[i] > ret[guess]) guess = i;
			}
			for (int i = 0; i < SUBHEIGHT; i++)
				free(data[i]);
			free(data);
			free(ret);
			printf("**GUESS** %d\n", guess);
		} else if (strcmp(action, "validate") == 0) {
			//Run all training data
			int correct = 0;
			//for all digits
			for (int i = 0; i < 10; i++) {
				//for all variants of digit i
				for (int j = 80; j < 100; j++){
					if (j < 10) sprintf(inFile, "data/train/%d_0%d.pbm", i, j);
					else sprintf(inFile, "data/test/%d_%d.pbm", i, j);
					data = loadInput(inFile);
					double* ret = runNN(w1, w2, out, bias, data);
					// Find max
					int guess = 0;
					for (int i = 0; i < OUTPUT; i++) {
						if (ret[i] > ret[guess]) guess = i;
					}
					for (int i = 0; i < SUBHEIGHT; i++)
						free(data[i]);
					free(data);
					free(ret);
					//Tally correct number of guesses
					if (guess == i) correct++;
				}
			}
			if (trainCount >= 1600) printWeights(w1, w2, out, trainCount, bias);
			printf("%lf\n", correct/200.0); //200 test images
		} else if (strcmp(action, "test")) {
			// Not implemented yet
		} else {
			printf("Usage: <action> <inputFile>\n");
		}
	}
	//free everything
	if (fp != stdin) fclose(fp);
	for (int i = 0; i < HIDDEN1; i++)
		free(w1[i]);
	free(w1);
	for (int i = 0; i < HIDDEN2; i++)
		free(w2[i]);
	free(w2);
	for (int i = 0; i < OUTPUT; i++)
		free(out[i]);
	free(out);
	for (int i = 0;  i < BIAS; i++)
		free(bias[i]);
	free(bias);
	printf("done\n");
	return 0;
}

// Function to print the weight matrices to individual files when debugging
void printWeights(double** w1, double** w2, double** out, int count, double** bias) {
	char name[100];
	sprintf(name, "debug/%d_w1.txt", count);
	FILE* fp = fopen(name, "w");
	if (fp == NULL) {
		printf("Failed to open file\n");
		exit(0);
	}
	for (int row = 0; row < HIDDEN1; row++) {
		fprintf(fp, "[");
		for (int col = 0; col < INPUT; col++)
			fprintf(fp, "%lf, ", w1[row][col]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	sprintf(name, "debug/%d_w2.txt", count);
	fp = fopen(name, "w");
	if (fp == NULL) {
		printf("Failed to open file\n");
		exit(0);
	}
	for (int row = 0; row < HIDDEN2; row++) {
		for (int col = 0; col < HIDDEN1; col++)
			fprintf(fp, "%lf, ", w2[row][col]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	sprintf(name, "debug/%d_out.txt", count);
	fp = fopen(name, "w");
	if (fp == NULL) {
		printf("Failed to open file\n");
		exit(0);
	}
	for (int row = 0; row < OUTPUT; row++) {
		for (int col = 0; col < HIDDEN2; col++)
			fprintf(fp, "%lf, ", out[row][col]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	sprintf(name, "debug/%d_bias.txt", count);
	fp = fopen(name, "w");
	if (fp == NULL) {
		printf("Failed to open file\n");
		exit(0);
	}
	for (int col = 0; col < HIDDEN1; col++)
		fprintf(fp, "%lf, ", bias[0][col]);
	fprintf(fp, "\n");
	for (int col = 0; col < HIDDEN2; col++)
		fprintf(fp, "%lf, ", bias[1][col]);
	fprintf(fp, "\n");
	for (int col = 0; col < OUTPUT; col++)
		fprintf(fp, "%lf, ", bias[2][col]);
	fprintf(fp, "\n");
	fclose(fp);
}

//Applies the data to the NN
double* runNN(double** w1, double** w2, double** out, double** bias, int ** data) {
	double* result = malloc(OUTPUT*sizeof(double));
	// First matrix multiplication: Data*w1
	double * res1 = malloc(HIDDEN1*sizeof(double));
	double sum;
	for (int row = 0; row < HIDDEN1; row++) {
		sum = bias[0][row];
		for (int col = 0; col < INPUT; col++) {
			sum += data[col/SUBWIDTH][col%SUBWIDTH]*w1[row][col];
		}
		res1[row] = sum > 0 ? sum : 0;
	}
	// Second matrix multiplication: *w2
	double * res2 = malloc(HIDDEN2*sizeof(double));
	for (int row = 0; row < HIDDEN2; row++) {
		sum = bias[1][row];
		for (int col = 0; col < HIDDEN1; col++) {
			sum += res1[col]*w2[row][col];
		}
		res2[row] = sum > 0 ? sum : 0;
	}
	// 3rd Matrix multiplication: *out
	double* res3 = malloc(OUTPUT*sizeof(double));
	double total = 0;
	for (int row = 0; row < OUTPUT; row++) {
		sum = bias[2][row];
		for (int col = 0; col < HIDDEN2; col++) {
			sum += res2[col]*out[row][col];
		}
		res3[row] = exp(sum);
		total += res3[row];
	}
	// Apply softmax
	for (int i = 0; i < OUTPUT; i++) {
		result[i] = res3[i]/total;
	}
	free(res1);
	free(res2);
	free(res3);
	return result;
}

// Trains the NN
void trainNN(double** w1, double** w2, double** out, double** bias, int ** data, char expected, double learningRate) {

	double* result = malloc(OUTPUT*sizeof(double));
	double* result_e = malloc(OUTPUT*sizeof(double));
	double sum;

	// Find the derivatives of activation functions in the first hidden layer, whilst calculating the node values in res1
	double * res1 = malloc(HIDDEN1*sizeof(double));
	double * deriv1 = malloc(HIDDEN1*sizeof(double));
	for (int row = 0; row < HIDDEN1; row++) {
		sum = bias[0][row];
		for (int col = 0; col < INPUT; col++) {
			sum += data[col/SUBWIDTH][col%SUBWIDTH]*w1[row][col];
		}
		res1[row] = sum > 0 ? sum : 0;
		deriv1[row] = ((sum+H>0?sum+H:0)-(sum-H>0?sum-H:0))/(2*H);
	}
	
	// Find the derivatives of activation functions in the second hidden layer, whilst calculating the node values in res2
	double * res2 = malloc(HIDDEN2*sizeof(double));
	double * deriv2 = malloc(HIDDEN2*sizeof(double));
	for (int row = 0; row < HIDDEN2; row++) {
		sum = bias[1][row];
		for (int col = 0; col < HIDDEN1; col++) {
			sum += res1[col]*w2[row][col];
		}
		res2[row] = sum > 0 ? sum : 0;
		deriv2[row] = ((sum+H>0?sum+H:0)-(sum-H>0?sum-H:0))/(2*H);
	}
	
	// Find the derivatives of activation functions in the output layer, whilst calculating the node values in res3
	double * res3 = malloc(OUTPUT*sizeof(double));
	double * deriv3 = malloc(OUTPUT*sizeof(double));
	for (int row = 0; row < OUTPUT; row++) {
		sum = bias[2][row];
		for (int col = 0; col < HIDDEN2; col++) {
			sum += res2[col]*out[row][col];
		}
		res3[row] = sum;
	}
	//derivative of the softmax function is a little tricky because of the sum in the denominator, requires following for loop as well
	double total1, total2, total = 0;
	for (int i = 0; i < OUTPUT; i++) {
		total1 = 0;
		total2 = 0;
		for (int j = 0; j < OUTPUT; j++) {
			total1 += (j==i)?exp(res3[j]+H):exp(res3[j]);
			total2 += (j==i)?exp(res3[j]-H):exp(res3[j]);
		}
		deriv3[i] = (exp(res3[i]+H)/total1-exp(res3[i]-H)/total2)/(2*H);
		total += exp(res3[i]);
	}
	for (int i = 0; i < OUTPUT; i++) {
		result[i] = exp(res3[i])/total;
	}
	//Find error signals
	//calculate error in the output layer
	for (int i = 0; i < OUTPUT; i++)
		result_e[i] = (expected==i?1:0)-result[i];
	//calculate error in 2nd layer
	double* h2_e = calloc(HIDDEN2, sizeof(double));
	for (int row = 0; row < OUTPUT; row++) {
		for (int col = 0; col < HIDDEN2; col++)
			h2_e[col] += result_e[row]*out[row][col];
	}
	//calculate the error in the 1st layer
	double* h1_e = calloc(HIDDEN1, sizeof(double));
	for (int row = 0; row < HIDDEN2; row++) {
		for (int col = 0; col < HIDDEN1; col++)
			h1_e[col] += h2_e[row]*w2[row][col];
	}
	//Update all weights
	//Now update weights in w1
	for (int row = 0; row < HIDDEN1; row++) //for each node in the first hidden layer
		for (int col = 0; col < INPUT; col++)
			w1[row][col] += learningRate*h1_e[row]*deriv1[row]*data[col/SUBWIDTH][col%SUBWIDTH];
	//Also update bias weights for first layer
	for (int row = 0; row < HIDDEN1; row++)
		bias[0][row] += learningRate*deriv1[row];
	//Now update weights in w2
	for (int row = 0; row < HIDDEN2; row++) //for each node in the second hidden layer
		for (int col = 0; col < HIDDEN1; col++)
			w2[row][col] += learningRate*h2_e[row]*deriv2[row]*res1[col];
	//Also update bias weights for second layer
	for (int row = 0; row < HIDDEN2; row++)
		bias[1][row] += learningRate*deriv2[row];
	//Now update weights in out
	for (int row = 0; row < OUTPUT; row++) //for each node in the output layer
		for (int col = 0; col < HIDDEN2; col++)
			out[row][col] += learningRate*result_e[row]*deriv3[row]*res2[col];
	//Also update bias weights for output
	for (int row = 0; row < OUTPUT; row++)
		bias[2][row] += learningRate*deriv3[row];
	//I think we're done (phew) just free all the allocated shit
	free(result);
	free(result_e);
	free(h2_e);
	free(h1_e);
	free(res1);
	free(deriv1);
	free(res2);
	free(deriv2);
	free(res3);
	free(deriv3);
}

//Returns the subsampled 25*35 data. 25 cols, 35 rows. data[row][col]
int ** loadInput(char * inFile) {
	FILE* fp = fopen(inFile, "r");
	if (fp == NULL) {
		printf("Error loading file\n");
		exit(0);
	}
	//Check that the header is as expected
	char header[100];
	int width, height;
	fscanf(fp, "%s %d %d", header, &width, &height);
	// printf("%d %d\n", height, width);
	if (strcmp(header, "P1") != 0 || width != 50 || height != 70) {
		printf("Something fishy about the file header. Going for a nap\n");
		fclose(fp);
		exit(0);
	}
	// Read in the data. Note we want every second pixel and row. Inspired by 1511 17s1 ass1
	int ** data = malloc(SUBHEIGHT*sizeof(int*));
	for (int i = 0; i < SUBHEIGHT; i++)
		data[i] = malloc(SUBWIDTH*sizeof(int));
	int pixel = fgetc(fp);
	int row = 0, col = 0, oddRow = 0, oddCol = 0;
	while (pixel != EOF && row < HEIGHT) {
		if (pixel == '0' || pixel == '1') {
			if (!oddRow && !oddCol) data[row>>1][col>>1] = pixel == '1';
			col++;
			if (col == WIDTH) {
				row++;
				col = 0;
			}
		}
		pixel = fgetc(fp);
	}
	fclose(fp);
	return data;
}