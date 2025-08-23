import numpy as np
import copy, math
import matplotlib.pyplot as plt

def compute_huber_cost(X,y,w,b,delta = 1.0):
    
	m = len(X)
	y_pred = X.dot(w) + b
	error = y - y_pred

	cost = 0.0

	for e in error:
		if abs(e) <= delta:
			cost += 0.5*(e**2)
		else:
			cost += delta * (abs(e) - 0.5*delta)
	
	return cost/m

#This function return dj_db and dj_dw with penalities to high loss
def compute_huber_gradient(X,y,w,b,delta = 1.0):
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
    

def compute_gradient_descedent(X,y,w,b,alpha,num_iterat,delta = 1.0):

    w = w.copy()        #avoid modifying global w within function
    j_history = []      #An array that store cost J and w at each iterations       

    for i in range(num_iterat):
        dj_dw,dj_db =  compute_huber_gradient(X,y,w,b,delta)

        #Update parameters using w,b,alpha and gradient
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

         # Save cost J at each iteration
        j_history.append(compute_huber_cost(X, y, w , b,delta))

        if i % 100 == 0 or i == num_iterat-1:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e}")
 
    return w, b, j_history #return w and J,w history for graphing