import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#files import
from compute_gradient import compute_gradient_descedent
from compute_normalize import zscore_normalize_feature
from data_correlation import calcule_conf_int,calcule_pearson
from adam_correction import compute_adam

#Data import and tratment
try:
  df = pd.read_csv("Housing.csv")
  target_col = 'price'
    
  # Data preprocessing
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  x_train = df[numeric_cols].drop(target_col, axis=1)
  y_train = df[target_col].values  # convert to numpy array

  #data correlations
  correlation,p_valor = calcule_pearson(x_train,y_train)
  confid_interv = calcule_conf_int(x_train,y_train)
 
  # Implement normalization
  x_train_norm = zscore_normalize_feature(x_train)
  feature_names = x_train.columns
  y_train_norm = zscore_normalize_feature(y_train.reshape(-1,1)).flatten()    #first we transforme a vector 1D(n,) in 2D(n,1) after we reshape to 1D

  # Initialize parameters
  alpha = 0.005        #learning_rate
  num_iter = 10000
  b = 0
  w = np.zeros(x_train_norm.shape[1])  # initialize the weights for all features

  # Run gradient descent
  print("Starting gradient descent...")
  w_descedent, b_descendent, j_descedent = compute_gradient_descedent(
        x_train_norm, y_train_norm, w, b, alpha, num_iter
    )

  # Run the adam method
  print("Starting adam...")
  w_adam, b_adam, j_adam = compute_adam(
    x_train_norm,y_train_norm,w,b,alpha,num_iter
    )

  # Prediction with gradient descendent
  y_pred = x_train_norm @ w_descedent + b_descendent

  #Plotting the original data and regression line
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.scatter(x_train_norm[:, 0], y_train_norm, marker='x', c='r', label="Actual Value")
  plt.scatter(x_train_norm[:, 0], y_pred, color='blue', label="Predicted Value", alpha=0.5)
  plt.title(f"Feature vs price (first feature: {feature_names[0]})")
  plt.xlabel("Normalized " + feature_names[0])
  plt.ylabel("price")
  plt.grid(True)
  plt.legend()

  #Plotting cost function
  plt.subplot(1,2,2)
  plt.plot(j_descedent, label= "Gradient descendent")
  plt.plot(j_adam, label = "Adam")
  plt.title("cost history comparitions")
  plt.xlabel("Iteration")
  plt.ylabel("Cost")
  plt.grid(True)

  plt.tight_layout()
  plt.show()

  #Data visualizaton of x_train and y_train
  print("\n" + ".................................................." + "\n")
  print(f"X train shape: {x_train_norm.shape}.\t X train type: {type(x_train_norm)}\n")
  print(f"First 5 rows of X:\n{x_train.head()}\n")
  print(f"y train shape: {y_train_norm.shape}.\t y train type: {type(y_train_norm)}")
  print(f"First 5 y values: {y_train[:5]}\n")
  print(f"Final weights (Gradient): {w_descedent}")
  print(f"Final bias (Gradient): {b_descendent}")

  #Correlation visualization
  print("\n" + "..............................." + "\n")
  print(f"Correlation measure: {correlation}")
  print(f"p-valor: {p_valor}")
  print(f"Confident interval: {confid_interv}")
	
except Exception as e:
  print(f"An error occurred: {e}")

