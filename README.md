# Neural Network MNIST Classifier

A simple neural network implementation in Go for classifying handwritten digits from the MNIST dataset.

## Features

- 3-layer neural network (784 → 16 → 16 → 10)
- ReLU activation for hidden layers
- Softmax activation for output layer
- Cross-entropy loss function
- Model saving and loading functionality
- Command-line interface for training and prediction

## Usage

### Training the Model

To train the neural network on the MNIST dataset:

```bash
go run . train
```

This will:
- Train the model for 1000 epochs
- Display training progress every 10 epochs
- Save the trained model to `trained_model.json`

### Making Predictions

To predict the digit in a single image:

```bash
go run . predict <path_to_image>
```

Example:
```bash
go run . predict ./mnist_png/testing/5/10.png
```

This will:
- Load the trained model from `trained_model.json`
- Process the input image
- Display the predicted digit and confidence scores for all classes

## Requirements

- Go 1.16 or later
- MNIST PNG dataset in the `mnist_png` directory structure:
  ```
  mnist_png/
  ├── training/
  │   ├── 0/
  │   ├── 1/
  │   └── ... (digits 0-9)
  └── testing/
      ├── 0/
      ├── 1/
      └── ... (digits 0-9)
  ```

## Model Architecture

- **Input Layer**: 784 neurons (28×28 pixel images, flattened)
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 16 neurons with ReLU activation  
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

## Example Output

### Training
```
Epoch: 0, Average Loss: 2.3026, Learning Rate: 0.010000
Epoch: 10, Average Loss: 1.8234, Learning Rate: 0.009901
Epoch: 20, Average Loss: 1.5432, Learning Rate: 0.009804
...
Training complete
Model saved to trained_model.json
```

### Prediction
```
Image: ./mnist_png/training/5/1032.png
Predicted digit: 5
Confidence: 89.23%

All probabilities:
  Digit 0: 0.0012 (0.12%)
  Digit 1: 0.0034 (0.34%)
  Digit 2: 0.0089 (0.89%)
  Digit 3: 0.0156 (1.56%)
  Digit 4: 0.0234 (2.34%)
  Digit 5: 0.8923 (89.23%)
  Digit 6: 0.0178 (1.78%)
  Digit 7: 0.0234 (2.34%)
  Digit 8: 0.0089 (0.89%)
  Digit 9: 0.0051 (0.51%)
```

## Important Notes

- **Model Training**: You must train the model first before making predictions. The trained model is saved as `trained_model.json`.
- **Image Format**: Input images should be 28×28 pixel PNG files, similar to the MNIST dataset format.
- **Model Accuracy**: The accuracy depends on the training duration. With 1000 epochs, the model should achieve reasonable accuracy for digit classification.
- **Error Handling**: The program includes error handling for missing files and untrained models.
