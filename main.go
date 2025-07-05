package main

import (
	"fmt"
	"math"
)

type Neuron struct {
	weights     []float64
	bias        float64
	weightGrads []float64
	biasGrad    float64
	input       []float64
	output      float64
}

type Layer struct {
	neurons []Neuron
	inputs  []float64
	outputs []float64
}
func NewLayer(inputSize int, neuronsCount int) Layer {
	neurons := make([]Neuron, neuronsCount)
	for i := range neurons {
		weights := make([]float64, inputSize)
		for j := range weights {
			weights[j] = randFloat64(-0.5, 0.5)
		}
		neurons[i] = Neuron{weights: weights, bias: randFloat64(-0.5, 0.5)}
	}

	return Layer{neurons: neurons}
}

func ReLu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Softmax function - activation function for the output layer
// for each x in x = exp(x) / sum(exp(x))
func Softmax(outputScores []float64) []float64 {
	// Find the maximum logit for numerical stability.
	// This is needed because exp(z_i) grows very large, and can overflow for big logits.
	// By subtracting outputScore from each z_i, we shift the largest value to 0
	// and the rest become negative, which prevents overflow but keeps the ratios the same.
	// Example:
	// logits = [2.0, 1.0, 1000.0]
	// outputScore = 1000.0
	// shifted logits = [-998, -999, 0]
	// exp(0) = 1
	// exp(-998) and exp(-999) are ~0
	// so softmax still works correctly without overflow.
	outputScore := outputScores[0]
	for _, v := range outputScores {
		if v > outputScore {
			outputScore = v
		}
	}

	expSum := 0.0
	exps := make([]float64, len(outputScores))
	for i, v := range outputScores {
		e := math.Exp(v - outputScore)
		exps[i] = e
		expSum += e
	}

	for i := range exps {
		exps[i] /= expSum
	}

	return exps
}

// Loss function
func CrossEntropyLoss(probs []float64, label int) float64 {
	return -math.Log(probs[label] + 1e-15) // add small epsilon to avoid log(0)
}

func (l *Layer) Forward(inputs []float64) []float64 {
	l.inputs = inputs
	outputs := make([]float64, len(l.neurons))

	for i, neuron := range l.neurons {
		// res = w1 * x1 + w2 * x2 + ... + wN * xN + b
		sum := neuron.bias
		for j, w := range neuron.weights {
			sum += w * inputs[j]
		}
		output := ReLu(sum) // apply activation function
		outputs[i] = output

		l.neurons[i].output = outputs[i]
		l.neurons[i].input = inputs
	}

	l.outputs = outputs
	return outputs
}

func main() {
	inputs := readFile("./mnist_png/training/1/1002.png")

	layer1 := NewLayer(784, 16)
	layer2 := NewLayer(16, 16)
	outputLayer := NewLayer(16, 10)

	for epoch := 0; epoch < 1000; epoch++ {
		out1 := layer1.Forward(inputs)
		out2 := layer2.Forward(out1)
		final := outputLayer.Forward(out2)

		label := 1
		probs := Softmax(final)
		loss := CrossEntropyLoss(probs, label)

		if epoch%10 == 0 {
			fmt.Println("Epoch:", epoch, "Loss:", loss)
		}
	}
	fmt.Println("Training complete")

}

