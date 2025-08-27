import numpy as np
from kmeans import kmeans

# synthetic two-cluster dataset
def main():
    rng = np.random.default_rng(0)
    X1 = rng.normal(0, 0.5, (50, 2))
    X2 = rng.normal(5, 0.5, (50, 2))
    X = np.vstack([X1, X2])
    centroids, labels, inertia, n_iter, history = kmeans(X, 2, init='kmeans++', random_state=0)
    print('centroids:', centroids)
    print('inertia:', round(inertia, 4))
    print('iterations:', n_iter)

if __name__ == '__main__':
    main()
