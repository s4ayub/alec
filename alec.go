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
	Sizes int
	OutputLayer int
	Biases
	Weights
	Outputs
	Deltas
	Changes
	Errors
}

func NewAlec(bThresh float32, lRate float32, mMentum float32) *Alec { // This is my constructor to instantiate an Alec
	al := &Alec{
		BinaryThresh: bThresh,
		LearningRate: lRate,
		Momentum: mMentum,
	}

	biases ....
	weights ....

	return al
}

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
