package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
)

type BatchData struct {
	Images [][]float64
	Labels []int
}

// GetBatch loads a batch of images from the MNIST dataset
func GetBatch(batchSize int, dataPath string) (*BatchData, error) {
	batch := &BatchData{
		Images: make([][]float64, 0, batchSize),
		Labels: make([]int, 0, batchSize),
	}

	// validate data path
	if _, err := os.Stat(dataPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("data path %s does not exist", dataPath)
	}

	// Get all digit directories (0-9)
	digitDirs := make([]string, 10)
	for i := 0; i < 10; i++ {
		digitDirs[i] = filepath.Join(dataPath, strconv.Itoa(i))
	}

	// Load images randomly from all digit directories
	for len(batch.Images) < batchSize {
		// Randomly select a digit directory
		digitLabel := rand.Intn(10)
		digitDir := digitDirs[digitLabel]

		// Get all PNG files in this digit directory
		files, err := filepath.Glob(filepath.Join(digitDir, "*.png"))
		if err != nil {
			return nil, fmt.Errorf("error reading directory %s: %v", digitDir, err)
		}

		if len(files) == 0 {
			continue // Skip if no files found
		}

		// Randomly select a file from this digit directory
		selectedFile := files[rand.Intn(len(files))]

		// Load the image
		imageData := ReadImage(selectedFile)

		// Add to batch
		batch.Images = append(batch.Images, imageData)
		batch.Labels = append(batch.Labels, digitLabel)
	}

	return batch, nil
}
