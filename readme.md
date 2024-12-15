# Neural Network with NumPy - Project Overview

## How We Built It

This project leverages NumPy to create, train, and evaluate neural networks based on customizable configuration files. These files define:
- Network layers
- Input formats
- Training strategies

We use a combination of multiple neural networks to make predictions to be able to compare the results and choose the best model for each step of the training process. We are also able to train multiple models at the same time or a specific model at a time.

### Key Components:
- **Model Generator:** Builds neural networks from configuration files.
- **Training Script:** Trains specific or multiple neural networks.
- **Prediction Script:** Makes predictions using trained models and plots real-time training curves.
- **Training and Prediction are inside the same script**
- **Dataset Scripts:** Generate synthetic datasets for training and testing.
---

## How Parallel Processing Works

We utilize **CUDA from Numba** for parallel processing, enabling faster computations.

### Key Steps:
1. **Data Preparation:** We load and preprocess data before sending it to the GPU.
2. **Batch Computation:** We parallelize calculations across batches of data.
3. **Model Evaluation:** Training steps like forward propagation, loss calculation, and backpropagation run concurrently on GPU threads.

### How we deal with overfitting:
- **Early Stopping:** We monitor validation loss and stop training when it starts increasing.
- **Dropout Layers:** We include dropout layers to prevent overfitting.
- **Regularization:** We apply L1 and L2 regularization to penalize large weights.

## How It Works

1. **Configuration Loading:** The script reads network configurations from YAML/JSON files.
2. **Model Initialization:** Neural networks are constructed dynamically based on the configurations.
3. **Training Process:**
   - Choose a specific network or train all defined models.
   - Select hyperparameter optimization strategies:
     - **Cosine Annealing**
     - **Cyclic Learning Rate**
     - **Plateau Reduction**
   - Monitor training progress in real-time with live plots.
4. **Evaluation and Predictions:** The trained models are evaluated against test datasets, generating accuracy, loss statistics, and prediction graphs.

## Graphs & Benchmarking

### Benchmark Results:
We benchmarked the project by comparing CPU-only vs GPU-accelerated runs.

**Results:**
- **Training Time Comparison:** GPU processing significantly reduced training times.
- **Accuracy Trends:** Consistent model improvement across multiple training sessions.

### Example Graphs:

![Training Curve](benchmarking\img\new_archi.png)
![Learning Rate Comparison](benchmarking\img\Balanced new cycle.png)

---

**Note:** Ensure CUDA and Numba are properly installed to benefit from GPU acceleration.
