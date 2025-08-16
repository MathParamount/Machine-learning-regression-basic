import numpy as np
from scipy.stats import stats
from scipy.stats import pearsonr

def calcule_conf_int(x_train,y_train):
	
	sample = np.array([x_train])
	
	mean = np.mean(sample)
		
	std = np.std(sample,ddof=1)		#standard-deviation

	#freedom degree
	fr_deg = len(sample) - 1

	#confident interval
	conf_int = stats.t.interval(0.95,fr_deg,loc = media,scale = (std/np.sqrt(fr_deg + 1)))

	print(f"confident interval of 95%: {conf_int}")


def calcule_pearson:

	correlation,p_valor = pearsonr(x_train[:0],y_train)

	print(f"Pearson confident: {correlatino}")
	print(f"p-valor: {p_valor}")
