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

#This function return dj_db and dj_dw
def compute_gradient_function_huber_loss(X, y, w, b, delta=1.0):

	m = len(y)
	y_pred = X.dot(w) + b

	error = y - (np.dot(X,w) + b)
	abs_error = np.abs(error)

	grad = np.where(abs_error <= delta, -error, -delta * np.sign(error))

	dj_dw = (1 / m) * np.dot(X.T, grad)
	dj_db = (1 / m) * np.sum(grad)

	return dj_dw, dj_db

def compute_adam(X,y,w,b,learn_rate,num_iter,delta=1.0):
	#initializations
	m_w,m_b = 0, 0
	v_w,v_b = 0, 0
	step = 0
	cost_history = []

	epsilon = 1e-8		#Adjust to avoid zero division
	Beta1 = 0.9			#Smooths the gradients
	Beta2 = 0.999		#Smooths the quadratic gradient

	for i in range(num_iter):
		step += 1
		
		#Getting dw,db and making a cost history
		dw , db = compute_gradient_function_huber_loss(X,y,w,b,delta)
		cost = compute_huber_cost(X,y,w,b,delta)
		
		cost_history.append(cost)
				
		#Exponential moment atualizations
		m_w = Beta1*m_w + (1-Beta1)*dw
		v_w = Beta2*v_w + (1-Beta2)*(dw**2)

		m_b = Beta1*m_b + (1-Beta1)*db
		v_b = Beta2*v_b + (1-Beta2)*(db**2)
		
		#Bias correction
		m_w_correction = m_w/(1-Beta1**step)
		m_b_correction = m_b/(1-Beta1**step)

		v_w_correction = v_w/(1-Beta2**step)
		v_b_correction = v_b/(1-Beta2**step)
		
		#Updating the parameters
		w = w - learn_rate*(m_w_correction/ (np.sqrt(v_w_correction)+epsilon))
		b = b - learn_rate*(m_b_correction/ (np.sqrt(v_b_correction)+epsilon))
		
	return w,b,cost_history
