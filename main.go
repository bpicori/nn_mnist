package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

type Neuron struct {
	Weights     []float64 `json:"weights"`
	Bias        float64   `json:"bias"`
	WeightGrads []float64 `json:"weight_grads"`
	BiasGrad    float64   `json:"bias_grad"`
	Input       []float64 `json:"input"`
	Output      float64   `json:"output"`
}

type Layer struct {
	Neurons []Neuron  `json:"neurons"`
	Inputs  []float64 `json:"inputs"`
	Outputs []float64 `json:"outputs"`
}

func NewLayer(inputSize int, neuronsCount int) Layer {
	neurons := make([]Neuron, neuronsCount)
	// Xavier/Glorot initialization for better gradient flow
	scale := math.Sqrt(2.0 / float64(inputSize))

	for i := range neurons {
		weights := make([]float64, inputSize)
		weightGrads := make([]float64, inputSize)
		for j := range weights {
			weights[j] = RandomFloat64(-scale, scale)
		}
		neurons[i] = Neuron{Weights: weights, Bias: 0.0, WeightGrads: weightGrads, BiasGrad: 0.0}
	}

	return Layer{Neurons: neurons}
}

func (l *Layer) Forward(inputs []float64) []float64 {
	l.Inputs = inputs
	outputs := make([]float64, len(l.Neurons))

	for i, neuron := range l.Neurons {
		// res = w1 * x1 + w2 * x2 + ... + wN * xN + b
		sum := neuron.Bias
		for j, w := range neuron.Weights {
			sum += w * inputs[j]
		}
		output := ReLu(sum) // apply activation function
		outputs[i] = output

		l.Neurons[i].Output = sum // Store pre-activation value
		l.Neurons[i].Input = inputs
	}

	l.Outputs = outputs
	return outputs
}

func (l *Layer) Backward(dLoss_Output []float64) []float64 {
	dLoss_Inputs := make([]float64, len(l.Inputs))

	for i, neuron := range l.Neurons {
		dReLU := 0.0
		if neuron.Output > 0 {
			dReLU = 1.0
		}
		dLoss_dZ := dLoss_Output[i] * dReLU

		for j := range neuron.Weights {
			gradW := dLoss_dZ * neuron.Input[j] // calculate gradient for weight
			// as we are using batches, we accumulate the gradient 
			// so then we can average it and update the weights with the average gradient
			neuron.WeightGrads[j] += gradW
			dLoss_Inputs[j] += dLoss_dZ * neuron.Weights[j]
		}

		neuron.BiasGrad += dLoss_dZ // see comment above
		l.Neurons[i] = neuron
	}

	return dLoss_Inputs
}

func (l *Layer) UpdateWeights(learningRate float64) {
	for i, neuron := range l.Neurons {
		for j := range neuron.Weights {
			neuron.Weights[j] -= learningRate * neuron.WeightGrads[j]
			neuron.WeightGrads[j] = 0 // Reset gradient after update
		}
		neuron.Bias -= learningRate * neuron.BiasGrad
		neuron.BiasGrad = 0 // Reset bias gradient
		l.Neurons[i] = neuron
	}
}

// Predict performs inference on a single image
func Predict(model *Model, imageData []float64) (int, []float64) {
	// Forward pass through the network
	out1 := model.Layer1.Forward(imageData)
	out2 := model.Layer2.Forward(out1)
	final := model.OutputLayer.Forward(out2)

	// Apply softmax to get probabilities
	probs := Softmax(final)

	// Find the class with highest probability
	maxProb := 0.0
	predictedClass := 0
	for i, prob := range probs {
		if prob > maxProb {
			maxProb = prob
			predictedClass = i
		}
	}

	return predictedClass, probs
}

func TrainModel() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	layer1 := NewLayer(784, 128)
	layer2 := NewLayer(128, 128)
	outputLayer := NewLayer(128, 10)

	initialLearningRate := 0.01
	batchSize := 1000
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
			copy(dL_dZ, probs)
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

	// Save the trained model
	model := &Model{
		Layer1:      layer1,
		Layer2:      layer2,
		OutputLayer: outputLayer,
	}

	err := model.SaveModel("trained_model.json")
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
	} else {
		fmt.Println("Model saved to trained_model.json")
	}
}

func PredictSingleImage(imagePath string) {
	// Check if image file exists
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		fmt.Printf("Error: Image file '%s' does not exist\n", imagePath)
		return
	}

	// Load the trained model
	model, err := LoadModel("trained_model.json")
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		fmt.Println("Make sure you have trained the model first by running: go run . train")
		return
	}

	// Load and process the image
	imageData := ReadImage(imagePath)

	// Make prediction
	predictedClass, probabilities := Predict(model, imageData)

	// Display results
	fmt.Printf("Image: %s\n", imagePath)
	fmt.Printf("Predicted digit: %d\n", predictedClass)
	fmt.Printf("Confidence: %.2f%%\n", probabilities[predictedClass]*100)
	fmt.Println("\nAll probabilities:")
	for i, prob := range probabilities {
		fmt.Printf("  Digit %d: %.4f (%.2f%%)\n", i, prob, prob*100)
	}
}

func TestModel() {
	// Load the trained model
	model, err := LoadModel("trained_model.json")
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		fmt.Println("Make sure you have trained the model first by running: go run . train")
		return
	}

	fmt.Println("Testing model on random images from test dataset...")

	// Test with random images from the test set
	testBatch, err := getBatch(500, "./mnist_png/test")
	if err != nil {
		fmt.Printf("Error loading test batch: %v\n", err)
		return
	}

	correct := 0
	total := len(testBatch.Images)

	// Track per-digit accuracy
	digitCorrect := make([]int, 10)
	digitTotal := make([]int, 10)

	fmt.Printf("Testing on %d random images...\n\n", total)

	for i := 0; i < total; i++ {
		imageData := testBatch.Images[i]
		actualLabel := testBatch.Labels[i]

		// Make prediction
		predictedClass, probabilities := Predict(model, imageData)

		// Update statistics
		digitTotal[actualLabel]++
		if predictedClass == actualLabel {
			correct++
			digitCorrect[actualLabel]++
		}

		// Show first 10 predictions as examples
		if i < 10 {
			fmt.Printf("Test %d: Actual=%d, Predicted=%d, Confidence=%.2f%% %s\n",
				i+1, actualLabel, predictedClass, probabilities[predictedClass]*100,
				func() string {
					if predictedClass == actualLabel {
						return "✓"
					}
					return "✗"
				}())
		}
	}

	// Calculate and display overall accuracy
	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("\n=== Test Results ===\n")
	fmt.Printf("Overall Accuracy: %d/%d (%.2f%%)\n", correct, total, accuracy)

	// Display per-digit accuracy
	fmt.Println("\nPer-digit accuracy:")
	for digit := 0; digit < 10; digit++ {
		if digitTotal[digit] > 0 {
			digitAccuracy := float64(digitCorrect[digit]) / float64(digitTotal[digit]) * 100
			fmt.Printf("  Digit %d: %d/%d (%.2f%%)\n", digit, digitCorrect[digit], digitTotal[digit], digitAccuracy)
		} else {
			fmt.Printf("  Digit %d: No test samples\n", digit)
		}
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage:")
		fmt.Println("  nn_mnist train                    - Train the model")
		fmt.Println("  nn_mnist predict <image_path>     - Predict digit from image")
		fmt.Println("  nn_mnist test                     - Test model accuracy on random test images")
		return
	}

	command := os.Args[1]

	switch command {
	case "train":
		TrainModel()
	case "predict":
		if len(os.Args) < 3 {
			fmt.Println("Error: Please provide an image path")
			fmt.Println("Usage: go run . predict <image_path>")
			return
		}
		imagePath := os.Args[2]
		PredictSingleImage(imagePath)
	case "test":
		TestModel()
	default:
		fmt.Printf("Unknown command: %s\n", command)
		fmt.Println("Available commands: train, predict, test")
	}
}
