import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from compute_gradient import compute_gradient_descedent
from compute_normalize import zscore_normalize_feature

#Data import and tratment
try:
  df = pd.read_csv("Housing.csv")
  target_col = 'price'
    
  # Data preprocessing
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  x_train = df[numeric_cols].drop(target_col, axis=1)
  y_train = df[target_col].values  # convert to numpy array

  # Implement normalization
  x_train_norm = zscore_normalize_feature(x_train)
  feature_names = x_train.columns

  # Initialize parameters
  alpha = 0.005  # learning_rate
  num_iter = 1000
  b = 0
  w = np.zeros(x_train_norm.shape[1])  # initialize the weights for all features

  # Run gradient descent
  print("Starting gradient descent...")
  w_final, b_final, j_history = compute_gradient_descedent(
        x_train_norm, y_train, w, b, alpha, num_iter
    )

  # Prediction
  y_pred = x_train_norm @ w_final + b_final


  #Plotting the original data and regression line
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.scatter(x_train_norm[:, 0], y_train, marker='x', c='r', label="Actual Value")
  plt.scatter(x_train_norm[:, 0], y_pred, color='blue', label="Predicted Value", alpha=0.5)
  plt.title(f"Feature vs price (first feature: {feature_names[0]})")
  plt.xlabel("Normalized " + feature_names[0])
  plt.ylabel("price")
  plt.grid(True)
  plt.legend()

  #Plotting cost function
  plt.subplot(1,2,2)
  plt.plot(j_history)
  plt.title("cost history")
  plt.xlabel("Iteration")
  plt.ylabel("Cost")
  plt.grid(True)

  plt.tight_layout()
  plt.show()

  #Data visualizaton of x_train and y_train
  print("\n" + ".................................................." + "\n")
  print(f"X train shape: {x_train_norm.shape}.\t X train type: {type(x_train_norm)}\n")
  print(f"First 5 rows of X:\n{x_train.head()}\n")
  print(f"y train shape: {y_train.shape}.\t y train type: {type(y_train)}")
  print(f"First 5 y values: {y_train[:5]}\n")
  print(f"Final weights: {w_final}")
  print(f"Final bias: {b_final}")

except Exception as e:
  print(f"An error occurred: {e}")

