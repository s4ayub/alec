package alec_test

import (
	"fmt"
	"math"
	"testing"
	"github.com/s4ayub/alec"
)

func TestNOR(t *testing.T) {
	a := alec.Build(0.2, 50000, 5)
	assert.Equal(t, a.LearningRate, 0.2, "Learning rate should be this value")
	assert.Equal(t, a.HiddenNeurons, 5, "Number of hidden neurons should be this value")
	assert.Equal(t, a.NumOfIterations, 50000, "Number of iterations should be this number")

	a.Train([][][]float64{ // NOR truth table
		{{0, 0}, {1}},
		{{0, 1}, {0}},
		{{1, 0}, {0}},
		{{1, 1}, {0}},
	})
	
	input := [][]float64{{0, 0}}
	guess := a.Smart(input)
	fmt.Println("NOR Test: ", guess.At(0, 0))
}

func TestXOR(t *testing.T) {
	a := alec.Build(0.2, 50000, 5)
	assert.Equal(t, a.LearningRate, 0.2, "Learning rate should be this value")
	assert.Equal(t, a.HiddenNeurons, 5, "Number of hidden neurons should be this value")
	assert.Equal(t, a.NumOfIterations, 50000, "Number of iterations should be this number")

	a.Train([][][]float64{ // NOR truth table
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	})
	
	input := [][]float64{{0, 0}}
	guess := a.Smart(input)
	fmt.Println("XOR TEST: ", guess.At(0, 0))
}

// Uncomment the following if you want to see it approxiamate the sine function between 0 -> 3.14
// func TestSIN(t *testing.T) {
// 	a := alec.Build(0.01, 1000000, 20)
// 	// assert.Equal(t, a.LearningRate, 0.05, "Learning rate should be this value")
// 	// assert.Equal(t, a.HiddenNeurons, 15, "Number of hidden neurons should be this value")
// 	// assert.Equal(t, a.NumOfIterations, 100000, "Number of iterations should be this number")

// 	a.Train([][][]float64{ // NOR truth table
// 		{{0}, {math.Sin(0)}}, // o to pi from here
// 		{{0.523599}, {math.Sin(0.523599)}},
// 		{{0.785398}, {math.Sin(0.785398)}},
// 		{{1.0472}, {math.Sin(1.0472)}},
// 		{{1.5708}, {math.Sin(1.5708)}},
// 		{{2.0944}, {math.Sin(2.0944)}},
// 		{{2.35619}, {math.Sin(2.35619)}},
// 		{{3.14159}, {math.Sin(3.14159)}},
// 		{{1.0}, {math.Sin(1.0)}}, // Random from here
// 		{{1.1}, {math.Sin(1.1)}},
// 		{{1.2}, {math.Sin(1.2)}},
// 		{{1.3}, {math.Sin(1.3)}},
// 		{{1.4}, {math.Sin(1.4)}},
// 		{{1.5}, {math.Sin(1.5)}},
// 		{{1.6}, {math.Sin(1.6)}},
// 		{{1.7}, {math.Sin(1.7)}},
// 		{{1.8}, {math.Sin(1.8)}},
// 		{{1.9}, {math.Sin(1.9)}},
// 		{{2.0}, {math.Sin(2.0)}},
// 		{{2.1}, {math.Sin(2.1)}},
// 		{{2.2}, {math.Sin(2.2)}},
// 		{{2.3}, {math.Sin(2.3)}},
// 		{{2.4}, {math.Sin(2.4)}},
// 		{{2.5}, {math.Sin(2.5)}},
// 		{{2.6}, {math.Sin(2.6)}},
// 		{{2.7}, {math.Sin(2.7)}},
// 		{{2.8}, {math.Sin(2.8)}},
// 		{{2.9}, {math.Sin(2.9)}},
// 		{{3.0}, {math.Sin(3.0)}},
// 	})
	
// 	numCorrect := 0.00
// 	set := 314.00
// 	var difference float64

// 	for i:=0.00; i<3.14; i+=0.01{ 
// 		input := [][]float64{{i}}
// 		guess := a.Smart(input)

// 		fmt.Println("guess: ", guess.At(0,0), "real: ", math.Sin(i))
// 		difference = math.Abs(guess.At(0,0)-math.Sin(i))
// 		// a neural network is correct when it consistently satisfies the parameters of it's design, in this case, 0.05 error margin when approximating values of sign
// 		if difference <= 0.05 { 
// 			numCorrect += 1.00
// 			fmt.Println("True")
// 		}
// 		fmt.Println("-------")
// 	}

// 	var percentCorrect float64
// 	percentCorrect = numCorrect/set
// 	fmt.Println("numcorrect:", numCorrect)
// 	fmt.Println("SIN TEST correctness: ", percentCorrect)
// }
