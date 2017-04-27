## Spect-Learn

Recognize digits from spectrograms generated from voice using convolutional neural network. The same model used as MNIST dataset. [Link](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)

### Don't forget to install the requirements
```pip install -r requirements.txt```

### Instructions
Use `tensorflow-gpu` to run on gpu.

Don't move the script to another directory. The images will be read from `spect` directory. To configure, change the folder name in the source file.

### Configuration
To change the parameters please view the source file.

### Run
```python train.py```

### Output
The model will be saved in a json file `model.json`.

The weights will be saved in `model.h5`.
