package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
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
	// Xavier/Glorot initialization for better gradient flow
	scale := math.Sqrt(2.0 / float64(inputSize))

	for i := range neurons {
		weights := make([]float64, inputSize)
		weightGrads := make([]float64, inputSize)
		for j := range weights {
			weights[j] = randFloat64(-scale, scale)
		}
		neurons[i] = Neuron{weights: weights, bias: 0.0, weightGrads: weightGrads, biasGrad: 0.0}
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

		// Store the pre-activation value for backpropagation
		l.neurons[i].output = sum // Store pre-activation value
		l.neurons[i].input = inputs
	}

	l.outputs = outputs
	return outputs
}

func (l *Layer) Backward(dLoss_Output []float64) []float64 {
	dLoss_Inputs := make([]float64, len(l.inputs))

	for i, neuron := range l.neurons {
		dReLU := 0.0
		if neuron.output > 0 {
			dReLU = 1.0
		}
		dLoss_dZ := dLoss_Output[i] * dReLU

		for j := range neuron.weights {
			// Ensure j is within bounds of input
			if j < len(neuron.input) {
				gradW := dLoss_dZ * neuron.input[j]
				neuron.weightGrads[j] += gradW

				// Ensure j is within bounds of dLoss_Inputs
				if j < len(dLoss_Inputs) {
					dLoss_Inputs[j] += dLoss_dZ * neuron.weights[j]
				}
			}
		}

		neuron.biasGrad += dLoss_dZ
		l.neurons[i] = neuron
	}

	return dLoss_Inputs
}

func (l *Layer) UpdateWeights(learningRate float64) {
	for i, neuron := range l.neurons {
		for j := range neuron.weights {
			neuron.weights[j] -= learningRate * neuron.weightGrads[j]
			neuron.weightGrads[j] = 0 // Reset gradient after update
		}
		neuron.bias -= learningRate * neuron.biasGrad
		neuron.biasGrad = 0 // Reset bias gradient
		l.neurons[i] = neuron
	}
}

func main() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	layer1 := NewLayer(784, 16)
	layer2 := NewLayer(16, 16)
	outputLayer := NewLayer(16, 10)

	initialLearningRate := 0.01
	batchSize := 32
	epochs := 1000

	for epoch := 0; epoch < epochs; epoch++ {
		// Get a batch of training data
		batch, err := getBatch(batchSize, "./mnist_png/training")
		if err != nil {
			fmt.Printf("Error loading batch: %v\n", err)
			continue
		}

		totalLoss := 0.0

		// Learning rate decay
		learningRate := initialLearningRate / (1.0 + float64(epoch)*0.001)

		// Process each image in the batch
		for i := 0; i < len(batch.Images); i++ {
			inputs := batch.Images[i]
			label := batch.Labels[i]

			// Forward pass
			out1 := layer1.Forward(inputs)
			out2 := layer2.Forward(out1)
			final := outputLayer.Forward(out2)

			// Calculate loss
			probs := Softmax(final)
			loss := CrossEntropyLoss(probs, label)
			totalLoss += loss

			// Backward pass
			dL_dZ := make([]float64, len(probs))
			for j := range probs {
				dL_dZ[j] = probs[j]
			}
			dL_dZ[label] -= 1.0 // Subtract 1 from the true class

			gradOut := outputLayer.Backward(dL_dZ)
			gradOut = layer2.Backward(gradOut)
			_ = layer1.Backward(gradOut) // Use _ to avoid ineffectual assignment warning

			// Update weights immediately after each sample (SGD)
			layer1.UpdateWeights(learningRate)
			layer2.UpdateWeights(learningRate)
			outputLayer.UpdateWeights(learningRate)
		}

		// Calculate average loss for the batch
		avgLoss := totalLoss / float64(batchSize)

		// Print progress every 10 epochs
		if epoch%10 == 0 {
			fmt.Printf("Epoch: %d, Average Loss: %.4f, Learning Rate: %.6f\n", epoch, avgLoss, learningRate)
		}
	}
	fmt.Println("Training complete")

}
