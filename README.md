- Resume:

The goal of this project is to predict house prices using machine learning techniques applied to a small dataset, which may be affected by multicollinearity. This project is somewhat complex because it attempts to predict house prices based on features such as area and various house amenities that influence the prediction.

The project consists of three .py files and one dataset file named Housing.csv, all located in the same directory. The main file is the compute_regression that inherits of others files.

During preprocessing, gradient computation and normalization are performed to ensure the data distribution is symmetric, enabling a more accurate analysis. Normalization is done using the StandardScaler from the sklearn library, which scales the data approximately to a range between -1 and 5, where -1 represents values slightly below the mean and 5 represents values far from the mean.

At the end of the program run, the output includes the number of iterations, the final cost, the first five rows of the dataset, the five most expensive houses (based on the target variable), and the final weights and bias.

Additionally, the program generates graphs showing the trend of the predictions compared to the original data, as well as a plot displaying cost versus iterations.

- Methodologies:

Import the CSV file and specify the "price" column as the target variable for prediction.
Preprocess the data by separating features (X_train) and the target variable (y_train).
Correlation analyzes between area and price column was a step to get more informations about the data.
Normalize the features using sklearn’s StandardScaler.
Initialize the model parameters, which can be adjusted to observe how changes affect the data.

Compute gradient descent through the following steps:
1. Calculate the cost (quadratic error function).
2. Compute the gradient (partial derivatives).
3. Iteratively update parameters to obtain optimal weights (w), bias (b), and cost history (j_history).
4. Make predictions (y_pred) by applying the normalized features to the final parameters.
5. Plot the original data points and the regression line showing the relationship between area and price.
6. Plot the cost function values over iterations to analyze model convergence.

- Observations:

This project was created to apply knowledges to something too useful in society.
The alpha and number of iterations variables can be change to visualize how the model performs their predictions.

- compilation:

the all libraries that was used in this project:

'bash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from compute_gradient import compute_gradient_descedent
from compute_normalize import zscore_normalize_feature
'

the 3 files is in the same directory, thus it doesn't need to specify the path to the compile.
'bash
python3 compute_regression.py
'

- Conclusion:

This project was made to use the concepts learned from the supervised machine learning regression and classification of Andrew ng. This work was so useful to aplly the hole concepts learned in this course for some useful thing to society.
