package main

import (
	"image/png"
	"math/rand"
	"os"
)

func readFile(path string) []float64 {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	img, err := png.Decode(f)
	if err != nil {
		panic(err)
	}

	bounds := img.Bounds()
	result := []float64{}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA() // In GO, RGBA returns 16-bit values, so we need to convert it to 8-bit to get the actual color for each channel

			// We need to convert every pixel to gray-scale. This is a standard formula to covert RGB to gray-scale
			gray := 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8)
			normalized := gray / 255.0

			result = append(result, normalized)
		}
	}

	return result
}

func randFloat64(min, max float64) float64 {
	return min + (max-min)*rand.Float64()
}

