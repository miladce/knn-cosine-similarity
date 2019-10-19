import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DistributedCosineKnn:
	def __init__(self, k=3):
		self.k=k

	def fit(self, input_data, n_epoch=1):
	    idxs=[]
	    dists=[]
	    epochs = np.array_split(input_data,n_epoch)
	    for b in range(n_epoch):
	    	cosim = cosine_similarity(epochs[b], input_data)
	    	idx0=[(heapq.nlargest((self.k+1), range(len(i)), i.take)) for i in cosim]
	    	idxs.extend(idx0)
	    	dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
	    return np.array(idxs),np.array(dists)
