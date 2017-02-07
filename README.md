# alec
Neural Network implementation in Go - Documentation: https://godoc.org/github.com/s4ayub/alec

[![Build Status](https://travis-ci.org/s4ayub/alec.svg?branch=master)](https://travis-ci.org/s4ayub/alec)

## Motivation and Features
This neural network currently allows users to specify a learning rate, the number of iterations for training and the number of units in the hidden layer of the network. Training is faciliated using a backpropagation algorithm. It was made for the purpose of learning basic machine learning concepts under supervised learning. Using Go allowed me to gain more experience with the language before delving into more research intensive projects using it.

## Train and Predict

```go
import (
	"fmt"
	"github.com/stevenmiller888/go-mind"
}

func main() {
	// An alec instance with a learning rate of 0.2, 1,000,000 iterations and 10 units in hidden layer
	a := alec.Build(0.2, 1000000, 10) 

	a.Train([][][]float64{ // XOR truth table
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	})

	input := [][]float64{{0, 0}}
	guess := a.Smart(input)
	fmt.Println("XOR TEST: ", guess.At(0, 0))
}
```
## Correctness
A 65.2% correctness was determined for the network when approximating sin(x). This test can be found within the comments of "alec_test.go". Improvements could be made with the amount of training data provided. Also, only the sigmoid activation function is used which may not be the best for training for this purpose. For example, when determing correctness with XOR truth table predictions, the correctness would be 95%+ yet, this is not an accurate representation of the true correctness of the network.

## Improvements
- Allow for different activation functions
- Refactor to support multiple hidden layers
- Implement custom error threshold to facilitate more custom training

## References
Reading Material:
- [How to Build a Neural Network](http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/)
- [Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs)

The following repositories were referenced as well:
- [mind](https://github.com/stevenmiller888/mind)
- [cerebrum](https://github.com/irfansharif/cerebrum)
- [brain js](https://github.com/harthur/brain)
