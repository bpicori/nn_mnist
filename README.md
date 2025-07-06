# Neural Network MNIST Classifier

A simple neural network implementation in Go for classifying handwritten digits from the MNIST dataset.

## Network Architecture (The Structure)
Our network has a specific structure:
- **Input Layer**: 784 neurons (28×28 pixel image flattened into a 1D array)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

### Activation Functions Used
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)` 
- **Softmax**: `f(x) = exp(x) / sum(exp(x))` 

### Loss Function Used
- **Cross-Entropy Loss**: `-log(p)` where `p` is the predicted probability of the correct class

### Optimization Algorithm Used
- **Stochastic Gradient Descent (SGD)**: Updates weights after each training sample


### Forward Propagation Process
This is how the network makes predictions:
```
Input (784 pixels) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (128 neurons) → Output (10 probabilities)
```
Each neuron calculates: `output = activation_function(sum(weights × inputs) + bias)`

### Backpropagation Process
- **Cross-Entropy Loss**: Measures how wrong our predictions are compared to the actual labels
- **Backpropagation**: Uses calculus (chain rule) to calculate how much each weight contributed to the error, then adjusts weights to reduce future errors
- **Gradient Descent**: The optimization algorithm that updates weights in the direction that reduces the loss

### Training Flow 

1. **Initialize**: Create random weights for all neurons
2. **For each epoch (1000 times)**:
   - Load a batch of 1000 random training images
   - **For each image in the batch**:
     - **Forward Pass**: Run the image through the network to get a prediction
     - **Calculate Loss**: Compare prediction with actual label using cross-entropy
     - **Backward Pass**: Calculate gradients (how much each weight should change)
     - **Update Weights**: Adjust all weights using the calculated gradients
3. **Save Model**: Store the trained weights to `trained_model.json`

## Usage

### Download the MNIST Dataset
```bash
make download_dataset
```
It will download the dataset to `mnist-pngs` directory.

### Training the Model

To train the neural network on the MNIST dataset:

```bash
go build -o nn_mnist *.go && ./nn_mnist train
```

### Making Predictions

To predict the digit in a single image:

```bash
./nn_mnist predict <path_to_image>
```


### Testing Model Accuracy

To test the model's accuracy on random images from the test dataset:

```bash
go run . test
```

## Resources
- https://www.youtube.com/watch?v=VMj-3S1tku0&t=193s
- https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=339s
- [MNIST in PNG Format](https://github.com/rasbt/mnist-pngs)
