import numpy as np
import matplotlib.pyplot as plt


# RBF nets are used for regression or function approximation


def kmeans(X, k):
    """Performs k-means clustering for 1D input

        Arguments:
            X {ndarray} -- A Mx1 array of inputs
            k {int} -- Number of clusters

        Returns:
            ndarray -- A kx1 array of final cluster centers
    """
    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)

        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)

        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)

    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return clusters, stds


class RBFNet(object):
    def __init__(self, k, step, iterations, inferStds):

        self.k = k
        self.step = step
        self.iterations = iterations
        self.inferStds = inferStds

        self.weight = np.random.randn(k)
        self.bias = np.random.randn(1)

    def rbf(self, x, c, s):
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def fit(self, X, y, cost_vis):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)

        else:
            # use fixed std
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1, -c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # training the model
        for iter in range(self.iterations):
            for i in range(X.shape[0]):
                # forward pass

                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.weight) + self.bias

                loss = (y[i] - F).flatten() ** 2
                if cost_vis:
                    print(f"Cost: {loss[0]}")

                # backward pass
                error = -(y[i] - F).flatten()

                # update weights and biases
                self.weight = self.weight - self.step * a * error
                self.bias = self.bias - self.step * error

    def predict(self, X):
        pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.weight) + self.bias
            pred.append(F)
        return np.array(pred)


if __name__ == '__main__':
    # sample inputs and add noise
    NUM_SAMPLES = 100
    X = np.random.uniform(0., 1., NUM_SAMPLES)
    X = np.sort(X, axis=0)
    noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
    y = np.sin(2 * np.pi * X) + noise

    rbfnet = RBFNet(k=2, step=0.01, iterations=1000,inferStds=True)
    rbfnet.fit(X, y, cost_vis=True)

    y_pred = rbfnet.predict(X)
    print(y_pred)

    plt.plot(X, y, '-o', label='true')
    plt.plot(X, y_pred, '-o', label='RBF-Net')
    plt.legend()

    plt.tight_layout()
    plt.show()
