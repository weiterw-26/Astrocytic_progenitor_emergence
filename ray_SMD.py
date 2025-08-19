import numpy as np
from scipy.sparse import load_npz, save_npz, coo_matrix
import ray
from SMD import SMD, shuffle_data, loader
from TSMD import TSMD
import sys
from scipy.stats import zscore

def main():

	name = 'W9_sub_5000'
	iteration = ''
	kmax = 40
	k_prior_min = 5
	trials = 3000
         	

	normalize = True
	#X = loader(name, preprocess=False)
	#X = np.transpose(X)
	X = load_npz(f'{name}.npz')
	X = np.array(X.todense())
	#X = np.load(f'{name}.npy')
	#n_sub = int(X.shape[0]*.5)
	if normalize==True:

		X = np.log1p(X)
		std = np.std(X,axis=0)>0.02
		mean = np.mean(X,axis=0)>0.02
		filt = np.all([std, mean], axis=0)
		X = X[:, filt]
		X = X/np.std(X, axis=0) 	
	n_sub = int(.4*X.shape[0])
	n_sub = None

	ray.init()
	#save_npz('3mo_norm',coo_matrix(X) )
	assert ray.is_initialized()==True

	if trials=='auto':
		out  = TSMD(X,int(kmax)/2, trials=trials, n_sub=n_sub, cluster_prior_min=k_prior_min, Z_velocity = 0.08)
		g = out[0]
		print(out[1])
		sys.stdout.flush()
	else:
		g = SMD(X,int(kmax/2),trials = trials, n_sub = n_sub, cluster_prior_min = k_prior_min, class_algo='entropy')
		
	#Xs = shuffle_data(X)

	#gs = SMD(Xs,int(kmax)/2,trials = trials, n_sub = n_sub, cluster_prior_min = k_prior_min)

	#z = (g-np.mean(gs))/np.std(gs) 
	z = g
	np.save(f'z_{name}_{iteration}.npy', z)
	ray.shutdown()
	assert ray.is_initialized() == False	

if __name__ == '__main__':
	main()
