# Fashion-MNIST ANN (Multi-Class Classification)

A simple **Artificial Neural Network (ANN)** (fully-connected feed-forward network) for **multi-class image classification** on the **Fashion-MNIST** dataset.

This project is implemented in a single Jupyter Notebook: `fashion_mnist.ipynb`.

## Dataset

This notebook uses the Kaggle CSV version of Fashion-MNIST:
- Dataset: *Fashion MNIST (Zalando Research)*
- Link: https://www.kaggle.com/datasets/zalando-research/fashionmnist

The dataset contains grayscale 28×28 images (flattened to 784 features) across **10 classes**.

> Note: The notebook expects the CSV files to be present locally:
> - `fashion-mnist_train.csv`
> - `fashion-mnist_test.csv`

## What the notebook does

1. Loads `fashion-mnist_train.csv` and `fashion-mnist_test.csv`
2. Concatenates them into a single dataframe (`df`) and saves `fashion-mnist.csv`
3. Basic checks (shape, missing values) and simple visualization of a sample image
4. Splits data into train/test (80/20) using `train_test_split`
5. Normalizes pixel values by scaling to `[0, 1]` (dividing by 255)
6. Uses **Keras Tuner** (`keras_tuner.RandomSearch`) to search over:
   - number of dense layers (1–10)
   - units per layer
   - activation functions (`relu`, `sigmoid`, `tanh`)
   - optimizer (`sgd`, `rmsprop`, `adam`, `adagrad`, `adadelta`, `adamax`, `nadam`, `ftrl`)
7. Trains the best model with **EarlyStopping** (monitoring `val_loss`)
8. Evaluates on the test set and runs a single-sample prediction visualization

## Model

The tuned model is a `keras.Sequential` network of Dense layers followed by a 10-unit softmax output.

- Input: 784-dimensional vector (flattened 28×28 image)
- Output: 10 classes (softmax)
- Loss: `sparse_categorical_crossentropy`
- Metric: accuracy

## Results

From the notebook run:
- **Validation accuracy during tuning:** best `val_accuracy` ≈ **0.8723** (RandomSearch over 10 trials)
- **Test accuracy:** **~0.8786**

(Exact values may vary depending on random seeds, environment, and tuner results.)

## Requirements

You can run this in **Google Colab** or locally.

Typical requirements:
- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow / keras
- keras-tuner

Install (example):

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras keras-tuner
```

## How to run

1. Download the dataset from Kaggle (link above)
2. Place the CSV files in the repository root (same folder as the notebook):
   - `fashion-mnist_train.csv`
   - `fashion-mnist_test.csv`
3. Open and run the notebook:

```bash
jupyter notebook fashion_mnist.ipynb
```

Or in Google Colab:
- Upload the notebook and the dataset CSVs
- Run all cells

## Notes / Tips

- The notebook concatenates train and test into a single CSV (`fashion-mnist.csv`). You can skip that step if you prefer.
- Training uses `EarlyStopping` to prevent overfitting.
- Hyperparameter tuning is kept small (`max_trials=10`) to run quickly.

## Acknowledgements

- Fashion-MNIST dataset by Zalando Research.
- Kaggle dataset page: https://www.kaggle.com/datasets/zalando-research/fashionmnist
