package main

import (
	"encoding/json"
	"os"
)

type Model struct {
	Layer1      Layer `json:"layer1"`
	Layer2      Layer `json:"layer2"`
	OutputLayer Layer `json:"output_layer"`
}

// SaveModel saves the trained model to a JSON file
func (m *Model) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(m)
}

// LoadModel loads a trained model from a JSON file
func LoadModel(filename string) (*Model, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var model Model
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

