import numpy as np
import copy, math
import matplotlib.pyplot as plt

def compute_cost(X,y,w,b):
    
    m = X.shape[0]
    
    prediction = np.dot(X, w) + b      #matrix multiplication

    cost =  (1/(2*m))*np.sum(prediction-y)**2
    
    return cost

#This function return dj_db and dj_dw
def compute_gradient_function(X, y, w, b):

    m = X.shape[0]
    error = (np.dot(X, w) + b) - y
    
    dj_dw = (1/m) * np.dot(X.T,error)
    dj_db = (1/m) * np.sum(error)

    return dj_dw,dj_db

def compute_gradient_descedent(X,y,w,b,alpha,num_iterat):

    w = copy.deepcopy(w)        #avoid modifying global w within function
    j_history = []          #An array that store cost J and w at each iterations       

    for i in range(num_iterat):
        dj_dw,dj_db =  compute_gradient_function(X,y,w,b)

        #Update parameters using w,b,alpha and gradient
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

         # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            j_history.append( compute_cost(X, y, w , b))

        if i % 100 == 0 or i == num_iterat-1:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e}")
 
    return w, b, j_history #return w and J,w history for graphing