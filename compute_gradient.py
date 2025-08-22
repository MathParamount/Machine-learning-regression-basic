import numpy as np
import copy, math
import matplotlib.pyplot as plt

def compute_cost(X,y,w,b):
    
    m = X.shape[0]
    
    prediction = np.dot(X, w) + b      #matrix multiplication

    cost =  (1/(2*m))*np.sum(prediction-y)**2
    
    return cost

#This function return dj_db and dj_dw with penalities to high loss
def compute_huber_loss(X,y,w,b,delta = 1.5):
    m = len(X)
    y_pred = X.dot(w) + b

    erro = y - y_pred

    dj_dw = np.zeros_like(w)
    dj_db = 0.0

    for i in range(m):
        e = erro[i]

        if abs(e) <= delta:
            grad = -e

        else:
            grad = -delta*np.sign(e)
        
        dj_dw += grad*X[i]
        dj_db += grad

        dj_dw = dj_dw/m
        dj_db = dj_db/m

        return dj_dw,dj_db
    

def compute_gradient_descedent(X,y,w,b,alpha,num_iterat,delta = 1.5):

    w = w.copy()        #avoid modifying global w within function
    j_history = []          #An array that store cost J and w at each iterations       

    for i in range(num_iterat):
        dj_dw,dj_db =  compute_huber_loss(X,y,w,b,delta)

        #Update parameters using w,b,alpha and gradient
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

         # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            j_history.append( compute_cost(X, y, w , b))

        if i % 100 == 0 or i == num_iterat-1:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e}")
 
    return w, b, j_history #return w and J,w history for graphing