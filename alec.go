package alec

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

// Use gonum/matrix library for matrix addition, subtraction, mul, dot product, etc.

type Alec struct {
	hiddenLayers float32
	learningRate float32
	iterations float32
	hiddenNeurons float32
}

