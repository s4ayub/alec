// Basic neural network
package alec

import (
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"time"
)

// Use gonum/matrix library for matrix addition, subtraction, mul, dot product, etc.
// mat64.Dense is in the form of a matrix

type Alec struct {
	LearningRate        float64
	NumOfIterations     int
	HiddenNeurons       int
	InputSynapses       *mat64.Dense // Connections between neurons
	OutputSynapses      *mat64.Dense // Connections between neurons
	HiddenNeuronSum     *mat64.Dense // Sums and results coming in and out of each neuron
	HiddenNeuronResults *mat64.Dense
	NeuronOutputSum     *mat64.Dense
	NeuronOutputResults *mat64.Dense
}

// Helper to grab data from the 3D array of training data
func Scrape(trainingData [][][]float64) (*mat64.Dense, *mat64.Dense) { // Training data is in the form of a 3D array
	var inputData, outputData []float64 // Slices of the input and output data

	numOfRows := len(trainingData)           // This is the length of the outer array
	inputColumns := len(trainingData[0][0])  // Length of the input columns
	outputColumns := len(trainingData[0][1]) // Length of output columns, all the inputs and outputs will share this length

	for _, dataSet := range trainingData { // Blank identifier allows me to drop the first return value (think key, value for loops in python)
		inputData = append(inputData, dataSet[0]...) // The ... is the syntax for appending one slice to another
		outputData = append(outputData, dataSet[1]...)
	}

	inputMatrix := mat64.NewDense(numOfRows, inputColumns, inputData)
	outputMatrix := mat64.NewDense(numOfRows, outputColumns, outputData)

	return inputMatrix, outputMatrix
}

// Helper to randomize weights of synapses
func Randoms(rows, columns int) *mat64.Dense {
	rand.Seed(time.Now().UTC().UnixNano()) // Set the starting point for a random number generator
	randomMatrix := mat64.NewDense(rows, columns, nil)

	for row := 0; row < rows; row++ {
		for col := 0; col < columns; col++ {
			randomMatrix.Set(row, col, rand.NormFloat64())
		}
	}

	return randomMatrix
}

// Helper function that applies the sigmoid function to a value
func Sigmoid(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

// Helper function that applies the derivative of the sigmiod function to a value
func SigmoidPrime(value float64) float64 {
	return Sigmoid(value) * (1 - Sigmoid(value))
}

// Activation function -> currently only uses sigmoid function
func SigmoidActivate(m *mat64.Dense) *mat64.Dense {
	rows, columns := m.Dims()
	activatedMatrix := mat64.NewDense(rows, columns, nil)

	for row := 0; row < rows; row++ {
		for col := 0; col < columns; col++ {
			neuronValue := m.At(row, col)
			activatedMatrix.Set(row, col, Sigmoid(neuronValue))
		}
	}

	return activatedMatrix
}

func SigmoidPrimeActivate(m *mat64.Dense) *mat64.Dense {
	rows, columns := m.Dims()
	activatedMatrix := mat64.NewDense(rows, columns, nil)

	for row := 0; row < rows; row++ {
		for col := 0; col < columns; col++ {
			neuronValue := m.At(row, col)
			activatedMatrix.Set(row, col, SigmoidPrime(neuronValue))
		}
	}

	return activatedMatrix
}

// Helper function to use forward propagation on the network
func (a *Alec) ForwardPropagate(inputMatrix *mat64.Dense) {
	// Propagating from input layer to hidden layer
	HiddenNeuronSum := &mat64.Dense{}
	HiddenNeuronSum.Mul(inputMatrix, a.InputSynapses)        // Matrix multiply input data through the weights
	a.HiddenNeuronResults = SigmoidActivate(HiddenNeuronSum) // Apply sigmoid activation function to sum at each layer

	// Propagating from hidden layer to output layer
	NeuronOutputSum := &mat64.Dense{}
	NeuronOutputSum.Mul(a.HiddenNeuronResults, a.OutputSynapses)
	a.NeuronOutputResults = SigmoidActivate(NeuronOutputSum)

	a.HiddenNeuronSum = HiddenNeuronSum // Keeping track of the sums, before activation function at each neuron
	a.NeuronOutputSum = NeuronOutputSum
}

// The following process is based off a formula used to determine the changes
// in weights required at each layer. These changes in weights are then added
// to the original weights to calibrate the network. The formula deals with
// the derivative of the activation function and applying it to the changes.
func (a *Alec) BackPropagate(inputMatrix *mat64.Dense, outputMatrix *mat64.Dense) {
	OutputLayerError := &mat64.Dense{} // Margin of error between actual and predicted results
	OutputLayerDelta := &mat64.Dense{} // The amount that we have to step back

	InputHiddenChanges := &mat64.Dense{}  // Extract the change in each weight from input layer to hidden layer
	OutputHiddenChanges := &mat64.Dense{} // Extract the change in each weight from output layer to hidden layer

	HiddenLayerDelta := &mat64.Dense{}

	// Matrix subtraction between training data and network's output results to get error margin
	OutputLayerError.Sub(outputMatrix, a.NeuronOutputResults)

	// Adjusting synapses and neurons from output layer to hidden layer
	OutputLayerDelta.MulElem(SigmoidPrimeActivate(a.NeuronOutputSum), OutputLayerError) // Multiplying by rate of change allows us to get the change of sum
	OutputHiddenChanges.Mul(a.HiddenNeuronResults.T(), OutputLayerDelta)                // T() transposes a matrix
	OutputHiddenChanges.Scale(a.LearningRate, OutputHiddenChanges)                      // Multiply matrix by a scaler
	a.OutputSynapses.Add(OutputHiddenChanges, a.OutputSynapses)

	// Adjusting synapses and neurons from hidden layer to input layer
	HiddenLayerDelta.Mul(OutputLayerDelta, a.OutputSynapses.T())
	HiddenLayerDelta.MulElem(SigmoidPrimeActivate(a.HiddenNeuronSum), HiddenLayerDelta)
	InputHiddenChanges.Mul(inputMatrix.T(), HiddenLayerDelta)
	InputHiddenChanges.Scale(a.LearningRate, InputHiddenChanges)
	a.InputSynapses.Add(InputHiddenChanges, a.InputSynapses)
}

// Initialize the network
func Build(learningRate float64, iterations int, hiddenNeurons int) *Alec { // Returns a pointer to Alec struct
	a := &Alec{}

	a.LearningRate = learningRate
	a.NumOfIterations = iterations
	a.HiddenNeurons = hiddenNeurons

	return a
}

func (a *Alec) Train(trainingData [][][]float64) { // Training data is in the form of a 3D array
	inputMatrix, outputMatrix := Scrape(trainingData)

	// We only need the number of columns because those represent the amount of layers at each level, in this case, input and output
	_, numInputColumns := inputMatrix.Dims() // Returns number of rows and cols in a matrix
	_, numOutputColumns := outputMatrix.Dims()

	// Prep our synapses matrixes
	a.InputSynapses = Randoms(numInputColumns, a.HiddenNeurons) // Randoms gives each synapse a random weight
	a.OutputSynapses = Randoms(a.HiddenNeurons, numOutputColumns)

	// Training network with forward and back propagation
	for i := 0; i < a.NumOfIterations; i++ {
		a.ForwardPropagate(inputMatrix)
		a.BackPropagate(inputMatrix, outputMatrix)
	}
}

// Ask a smart alec! This function allows you to make a prediction
func (a *Alec) Predict(inputData [][]float64) *mat64.Dense {
	var input []float64

	for _, data := range inputData {
		input = append(input, data...)
	}

	// Run the input data through the network
	inputMatrix := mat64.NewDense(len(inputData), len(inputData[0]), input)
	a.ForwardPropagate(inputMatrix)

	// Return the results from the output layer
	return a.NeuronOutputResults
}
