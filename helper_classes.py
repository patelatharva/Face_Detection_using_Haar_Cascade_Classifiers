import numpy as np


class WeakClassifier():
    """ weak classifier - threshold on the features
    Args:
        X (numpy.array): data array of flattened images
                        (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (num observations, )
    """
    def __init__(self, X, y, weights, thresh=0, feat=0, sign=1):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.idx_0  = self.ytrain == -1
        self.idx_1  = self.ytrain == 1
        self.threshold = thresh
        self.feature = feat
        self.sign = sign
        self.weights = weights

    def train(self):
        # save the threshold that leads to best prediction
        tmp_signs = []
        tmp_thresholds = []

        for f in range(self.Xtrain.shape[1]):
            m0 = self.Xtrain[self.idx_0, f].mean()
            m1 = self.Xtrain[self.idx_1, f].mean()
            tmp_signs.append(1 if m0 < m1 else -1)
            tmp_thresholds.append((m0+m1)/2.0)

        tmp_errors=[]
        for f in range(self.Xtrain.shape[1]):
            tmp_result = self.weights*(tmp_signs[f]*((self.Xtrain[:,f]>tmp_thresholds[f])*2-1) != self.ytrain)
            tmp_errors.append(sum(tmp_result))

        feat = tmp_errors.index(min(tmp_errors))

        self.feature = feat
        self.threshold = tmp_thresholds[feat]
        self.sign = tmp_signs[feat]
        # -- print self.feature, self.threshold

    def predict(self, x):
        return self.sign * ((x[self.feature] > self.threshold) * 2 - 1)