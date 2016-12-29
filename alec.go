package alec

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

// Use gonum/matrix library for matrix addition, subtraction, mul, dot product, etc.

type Alec struct {
	binaryThresh float32
	learningRate float32
	momentum float32
}

func NewAlec(bThresh float32, lRate float32, mMentum float32) *Alec { // This is my constructor to instantiate an Alec
	al := &Alec{
		binaryThresh: bThresh,
		learningRate: lRate,
		momentum: mMentum
	}
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
