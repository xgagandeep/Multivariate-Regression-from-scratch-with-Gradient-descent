
# Project: Multivariate Regression from Scratch with Gradient Descent

**Date:** 2020  
**Language:** Python  
**Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib  
**Type:** Multivariate Linear Regression

## Description

This project implements multivariate linear regression from scratch using gradient descent. It includes data normalization, hypothesis computation, error calculation, and gradient computation. The project demonstrates the entire process of building and training a linear regression model, as well as evaluating its performance.

## Features

- **Data Loading:** Utilizes the Boston housing dataset from Scikit-learn.
- **Data Normalization:** Standardizes feature values to improve model performance.
- **Linear Regression:** Implements multivariate linear regression with custom gradient descent.
- **Performance Evaluation:** Computes and plots R^2 score and error convergence.
- **Vectorization:** Enhances performance by vectorizing the gradient descent calculations.

## Files

- **`Multivariate-Regression-from-scratch-with-Gradient-descent.ipynb`:** Jupyter Notebook containing the implementation of multivariate linear regression from scratch.

## Installation

To run this project, you need Python and the required libraries installed. Follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/xgagandeep/Multivariate-Regression-from-scratch-with-Gradient-descenth.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Multivariate Regression from scratch with Gradient descent
   ```

3. **Install the required libraries:**

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

4. **Run the Jupyter Notebook:**

   ```bash
   jupyter notebook Multivariate Regression using gradient descent.ipynb
   ```

## Usage

1. **Load Data:** The project uses the Boston housing dataset from Scikit-learn.
2. **Normalize Data:** Standardizes the features for better model performance.
3. **Train Model:** Performs multivariate linear regression using gradient descent.
4. **Evaluate Model:** Calculates and plots the R^2 score and error convergence.
5. **Vectorization:** Includes optimized performance with vectorized operations.

## Functions

- `hypothesis(X, theta)`: Computes the predicted values based on the input features and model parameters.
- `error(X, y, theta)`: Calculates the mean squared error of the model.
- `gradient(X, y, theta)`: Computes the gradient of the error function.
- `gradientDescent(X, y, learning_rate, max_epochs)`: Optimizes model parameters using gradient descent.
- `r2_score(y, y_pred)`: Calculates the R^2 score to evaluate model performance.

## Performance

- **Vectorized Implementation:** Significantly improves performance and reduces computation time.

## Contribution

Feel free to contribute to this project by submitting issues or pull requests. For any questions or feedback, please open an issue on the GitHub repository.
