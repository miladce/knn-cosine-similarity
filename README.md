# distributed knn with cosine similarity (distance)
Cosine similarity is not a distance metric as it violates triangle inequality, and doesn’t work on negative data. and also, Scikit-learn's distance metrics doesn't have cosine distance. However, cosine similarity is fast, simple, and gets slightly better accuracy than other distance metrics on some datasets.

in this repository, (distributed) KNN algorithm implemented with cosine similarity.
and in this version, only the neighbors of each point and distance between them are specified. (not specify any label to data points)
</br></br>
## user guide
Parameters:	</br></br>
**k**:
Number of neighbors to use.</br></br>
**input_data**:
input data</br></br>
**n_epoch**:
By adjusting this parameter, your data split into smaller size sections. This parameter is useful in large datasets that you will get `MemoryError`.


### example:
```python
import extended_similarities as sims
import numpy as np

X= np.array([[-0.07,  0.31, -0.05],
       [-0.06,  0.27, -0.06],
       [-0.04,  0.3 , -0.04],
       [-0.06,  0.31, -0.  ],
       [-0.04,  0.29, -0.02],
       [-0.06,  0.3 , -0.01],
       [-0.05,  0.3 , -0.02],
       [-0.05,  0.31, -0.04],
       [-0.01,  0.3 , -0.04],
       [-0.08,  0.31, -0.05],
       [-0.03,  0.33, -0.02],
       [ 0.04,  0.34, -0.01],
       [-0.06,  0.3 , -0.02],
       [-0.14,  0.33, -0.01],
       [-0.15,  0.13, -0.08],
       [-0.16,  0.13, -0.08],
       [-0.05,  0.29, -0.03],
       [-0.05,  0.26, -0.01],
       [-0.04,  0.29, -0.04],
       [-0.05,  0.3 , -0.02],
       [-0.04,  0.3 , -0.02],
       [-0.07,  0.3 , -0.03],
       [-0.03,  0.29, -0.03],
       [-0.07,  0.26, -0.03],
       [-0.13,  0.25, -0.06],
       [-0.05,  0.28,  0.02],
       [-0.03,  0.31, -0.03],
       [ 0.02,  0.32, -0.04],
       [-0.04,  0.3 ,  0.  ],
       [-0.04,  0.29, -0.05],
       [-0.08,  0.29, -0.03],
       [-0.05,  0.31, -0.06],
       [ 0.04,  0.32, -0.02],
       [ 0.05,  0.33, -0.02]])

cos_knn = sims.DistributedCosineKnn(k=3)
indices, distances = cos_knn.fit(input_data=X, n_epoch=7)
```
