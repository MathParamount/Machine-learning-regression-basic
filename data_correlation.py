import numpy as np
from scipy.stats import t
from scipy.stats import pearsonr

def calcule_conf_int(x_train,y_train):
	
	sample = x_train.iloc[:,0].values
	
	mean = np.mean(sample)
		
	std = np.std(sample,ddof=1)		#standard-deviation

	#freedom degree
	fr_deg = len(sample) - 1

	#confident interval
	conf_int = t.interval(0.95,fr_deg,loc = mean,scale = (std/np.sqrt(fr_deg + 1)))

	return conf_int

def calcule_pearson(x_train,y_train):
	
	#The pearson correlation need to have just a single column
	correlation,p_valor = pearsonr(x_train.iloc[:,0],y_train)

	return correlation,p_valor
