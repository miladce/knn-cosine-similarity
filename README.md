# distributed knn with cosine similarity (distance)
Cosine similarity is not a distance metric as it violates triangle inequality, and doesn’t work on negative data. and also, Scikit-learn's distance metrics doesn't have cosine distance. However, cosine similarity is fast, simple, and gets slightly better accuracy than other distance metrics on some datasets.

in this repository, (distributed) KNN algorithm implemented with cosine similarity.
and in this version, only the neighbors of each point and distance between them are specified. (not specify any label to data points)
</br></br>
## user guide
Parameters:	</br></br>
**input_data**:
input data</br></br>
**k**:
Number of neighbors to use.</br></br>
**n_epoch**:
By adjusting this parameter, your data split into smaller size sections. This parameter is useful in large datasets that you will get `MemoryError`.
