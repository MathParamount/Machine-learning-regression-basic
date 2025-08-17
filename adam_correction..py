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

def compute_adam(X,y,w,b,learn_rate,num_iter):
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
		dw , db = compute_gradient_function(X,y,w,b)
		cost = compute_cost(X,y,w,b)
		cost_history.append(cost)
		
		#Exponential moment atualizations
		m_w = m_w*Beta1*(1-Beta1)*dw
		v_w = v_w*Beta2*(1-Beta2)*(dw**2)

		m_b = m_b*Beta1*(1-Beta1)*db
		v_b = v_b*Beta2*(1-Beta2)*(db**2)
		
		#Bias correction
		m_w_correction = m_w/(1-Beta1**step)
		m_b_correction = m_b/(1-Beta1**step)

		v_w_correction = v_w/(1-Beta2**step)
		v_b_correction = v_b/(1-Beta2**step)
		
		#Updating the parameters
		w = w - learn_rate*(m_w_correction/ (np.sqrt(v_w_correction)+epsilon))
		b = b - learn_rate*(v_b_correction/ (np.sqrt(v_b_correction)+epsilon))
		
	return w,b,cost_history
