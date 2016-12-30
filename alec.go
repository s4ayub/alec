package alec

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

// Use gonum/matrix library for matrix addition, subtraction, mul, dot product, etc.

type Alec struct {
	BinaryThresh float32
	LearningRate float32
	Momentum float32
	Sizes []int
	OutputLayer int
	Biases []int
	Weights []int
	Outputs []int
	Deltas []int
	Changes []int
	Errors []int
}

func (a Alec, mMentum float32, lRate float32, bThresh float32, sizers []int) { // This constructs the neural network
	a.BinaryThresh = bThresh
	a.LearningRate = lRate
	a.Momentum = mMentum
	a.sizes = sizers
	a.OutputLayer = len(a.sizes)

	for layer:= 0; layer < a.OutputLayer; layer++ {
		layer_size := a.sizes[layer]
		a.Deltas[layer] = zeros(layer_size)
		a.Errors[layer] = zeros(layer_size)
		a.Outputs[layer] = zeros(layer_size)
	}

}+

// need some structs
// need a constructor. something that instantiates an Alec
// need a training function, that takes the options of learning rate, and number of hidden layers
// need forward propagation
// need back propagation
// need prediction

	// hiddenLayers float32
	// learningRate float32
	// iterations float32
	// hiddenNeurons float32

// A bias allows you to shift the activation function left or right
