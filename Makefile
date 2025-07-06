build:
	go build -o nn_mnist *.go

download_dataset:
	git clone git@github.com:rasbt/mnist-pngs.git