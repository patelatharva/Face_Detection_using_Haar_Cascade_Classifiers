"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
import math
from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    X = np.ndarray(shape=(len(images_files), size[0] * size[1]))
    y = np.ndarray(shape=(len(images_files)))
    index = 0
    for image_file in images_files:
        label = int(image_file[7:9])
        image = cv2.imread(os.path.join(folder, image_file))
        gray_image = cv2.cvtColor(image, code=cv2.cv.CV_BGR2GRAY)
        scaled_image = cv2.resize(gray_image, dsize=(size[0],size[1]))
        flattened = scaled_image.flatten()
        X[index] = np.float32(flattened)
        y[index] = label
        index += 1
    return X, y


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    M = X.shape[0]
    N = int(p * M)
    permutation = np.random.permutation(M)
    Xtrain = X[permutation[0:N]]
    Xtest = X[permutation[N:]]
    ytrain = y[permutation[0:N]]
    ytest = y[permutation[N:]]
    return (Xtrain, ytrain, Xtest, ytest)


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    mean_face = get_mean_face(X)
    A = np.transpose(X - mean_face)
    C = np.dot(A,A.T)
    w, v = np.linalg.eigh(C)
    desc_w = w[::-1][:k]
    desc_v = v[:,::-1][:,:k]
    return (desc_v, desc_w)


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for iteration in range(self.num_iterations):
            self.weights /= sum(self.weights)
            h = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            h.train()
            self.weakClassifiers.append(h)
            mistaken_indices = []
            y_pred = np.zeros(shape=(self.Xtrain.shape[0]))
            for row in range(self.Xtrain.shape[0]):
                image = self.Xtrain[row]
                y_pred[row] = h.predict(image)
                if y_pred[row] != self.ytrain[row]:
                    mistaken_indices.append(row)
            e = sum(self.weights[mistaken_indices])
            alpha = 0.5*math.log((1-e)/e)
            self.alphas.append(alpha)
            if e > self.eps:
                for i in range(self.weights.shape[0]):
                    self.weights[i] = self.weights[i] * math.exp(-self.ytrain[i]*alpha*y_pred[i])
            else:
                break
    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        y_pred = self.predict(self.Xtrain)
        outcome = y_pred == self.ytrain
        correct = sum(outcome)
        incorrect = self.Xtrain.shape[0] - correct
        return (correct, incorrect)

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.alphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        results = np.zeros(shape=(X.shape[0]))
        for i in range(results.shape[0]):
            predictions = []
            for j in range(len(self.alphas)):
                y_pred = self.alphas[j] * self.weakClassifiers[j].predict(X[i])
                predictions.append(y_pred)
            results[i] = 1 if sum(predictions) > 0 else -1
        return results


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        image = np.zeros(shape=shape)
        feature_height = self.size[0]
        addition_height = feature_height/2
        subtraction_height = feature_height - addition_height
        feature_width = self.size[1]
        y = self.position[0]
        x = self.position[1]
        image[y:y+addition_height, x:x+feature_width] = 255
        image[y+addition_height:y + feature_height, x:x + feature_width] = 126
        return image

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape)
        feature_height = self.size[0]
        feature_width = self.size[1]
        addition_width = feature_width/2
        y = self.position[0]
        x = self.position[1]
        image[y : y+feature_height, x : x+addition_width] = 255
        image[y : y+feature_height, x + addition_width : x + feature_width] = 126
        return image


    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape)
        feature_height = self.size[0]
        feature_width = self.size[1]
        strip_height = feature_height/3
        y = self.position[0]
        x = self.position[1]
        image[y:y + strip_height, x:x + feature_width] = 255
        image[y + strip_height:y + 2*strip_height, x:x + feature_width] = 126
        image[y + 2*strip_height:y + feature_height, x:x + feature_width] = 255
        return image


    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape)
        feature_height = self.size[0]
        feature_width = self.size[1]
        strip_width = feature_width / 3
        y = self.position[0]
        x = self.position[1]
        image[y:y + feature_height, x:x + strip_width] = 255
        image[y:y + feature_height, x + strip_width:x + 2 * strip_width] = 126
        image[y:y + feature_height, x + 2*strip_width:x + feature_width] = 255
        return image


    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape)
        feature_height = self.size[0]
        feature_width = self.size[1]
        strip_width = feature_width / 2
        strip_height = feature_height/2
        y = self.position[0]
        x = self.position[1]

        #top left black
        image[y:y + strip_height, x:x + strip_width] = 126

        #top right white
        image[y:y + strip_height, x + strip_width:x + feature_width] = 255

        #bottom left white
        image[y + strip_height:y + feature_height, x:x + strip_width] = 255

        #bottom right black
        image[y + strip_height:y + feature_height, x + strip_width:x + feature_width] = 126
        return image


    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """


        def calc_sum(integral_image, area):
            position = area[0]
            size = area[1]
            extended_ii = cv2.copyMakeBorder(integral_image, top=1, bottom=0, left=1, right=0,
                                             borderType=cv2.BORDER_REPLICATE)
            mi_1 = (position[0], position[1])
            mi_2 = (position[0], position[1] + size[1])
            mi_3 = (position[0] + size[0], position[1])
            mi_4 = (position[0] + size[0], position[1] + size[1])
            sum_rect = extended_ii[mi_4[0], mi_4[1]] - extended_ii[mi_2[0],mi_2[1]] - extended_ii[mi_3[0], mi_3[1]] + extended_ii[mi_1[0], mi_1[1]]
            return sum_rect

        extended_ii = cv2.copyMakeBorder(ii, top=0, bottom=self.size[0], left=0, right=self.size[1],
                                         borderType=cv2.BORDER_CONSTANT, value=0)
        if self.feat_type == (2, 1):  # two_horizontal
            upper_add = (self.position,
                         (self.size[0]/2, self.size[1]))

            lower_sub = ((self.position[0] + self.size[0]/2, self.position[1]),
                         (self.size[0]/2, self.size[1]))
            score = calc_sum(extended_ii, upper_add) - calc_sum(extended_ii, lower_sub)


        if self.feat_type == (1, 2):  # two_vertical
            left_add = (
                self.position,
                (self.size[0], self.size[1]/2)
            )
            right_sub = (
                (self.position[0], self.position[1] + self.size[1]/2),
                (self.size[0], self.size[1]/2)
            )
            score = calc_sum(extended_ii, left_add) - calc_sum(extended_ii, right_sub)


        if self.feat_type == (3, 1):  # three_horizontal
            upper_add = (
                self.position,
                (self.size[0]/3, self.size[1])
            )
            middle_sub = (
                (self.position[0] + self.size[0]/3, self.position[1]),
                (self.size[0]/3, self.size[1])
            )
            lower_add = (
                (self.position[0] + 2*self.size[0]/3, self.position[1]),
                (self.size[0]/3, self.size[1])
            )
            score = calc_sum(extended_ii, upper_add) - calc_sum(extended_ii, middle_sub) + calc_sum(extended_ii, lower_add)
        if self.feat_type == (1, 3):  # three_vertical
            left_add = (
                self.position,
                (self.size[0], self.size[1]/3)
            )
            middle_sub = (
                (self.position[0], self.position[1] + self.size[1]/3),
                (self.size[0], self.size[1]/3)
            )
            right_add = (
                (self.position[0], self.position[1] + 2*self.size[1]/3),
                (self.size[0], self.size[1]/3)
            )
            score = calc_sum(extended_ii, left_add) - calc_sum(extended_ii, middle_sub) + calc_sum(extended_ii, right_add)
        if self.feat_type == (2, 2):  # four_square
            top_left_sub = (
                self.position,
                (self.size[0]/2, self.size[1]/2)
            )
            top_right_add = (
                (self.position[0], self.position[1] + self.size[1]/2),
                (self.size[0] / 2, self.size[1] / 2)
            )
            bottom_left_add = (
                (self.position[0] + self.size[0]/2, self.position[1]),
                (self.size[0] / 2, self.size[1] / 2)
            )
            bottom_right_sub = (
                (self.position[0] + self.size[0]/2, self.position[1] + self.size[1]/2),
                (self.size[0] / 2, self.size[1] / 2)
            )
            score = -calc_sum(extended_ii, top_left_sub) + calc_sum(extended_ii, top_right_add) + calc_sum(extended_ii, bottom_left_add) - calc_sum(extended_ii, bottom_right_sub)
        return score
def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    integral_images = []
    for image in images:
        integral_image = np.ndarray(shape=image.shape)
        integral_image[0,0] = image[0,0]
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                integral_image[y,x] = sum(sum(image[:y+1, :x+1]))
        integral_images.append(integral_image)
    return integral_images


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"
        for i in range(num_classifiers):

            # TODO: Complete the Viola Jones algorithm

            raise NotImplementedError

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        for x in scores:
            # TODO
            raise NotImplementedError

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        raise NotImplementedError
