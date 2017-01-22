package alec_test

import (
	"fmt"
	"testing"
	"github.com/s4ayub/alec"
	"github.com/stretchr/testify/assert"
)

func TestNOR(t *testing.T) {
	a := alec.Build(0.3, 20000, 5)
	assert.Equal(t, a.LearningRate, 0.3, "Learning rate should be this value")
	assert.Equal(t, a.HiddenNeurons, 5, "Number of hidden neurons should be this value")
	assert.Equal(t, a.NumOfIterations, 20000, "Number of iterations should be this number")

	a.Train([][][]float64{ // NOR truth table
		{{0, 0}, {1}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {0}},
		{{0, 0}, {1}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {0}},
		{{0, 0}, {1}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {0}},
		{{0, 0}, {1}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {0}},
	})
	
	input := [][]float64{{0, 0}}
	guess := a.Smart(input)
	fmt.Println(guess)
}