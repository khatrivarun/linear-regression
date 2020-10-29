import numpy as np


class LinearRegressor:
    def __init__(self, learning_rate=0.01, iterations=4000, x: np.ndarray = None, y: np.ndarray = None,
                 method='gradient'):
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        self.__X = x
        self.__Y = y
        self.__num_features = len(x)
        self.__weights = np.random.random(x.shape[1])
        self.__method = method

    """
    Function responsible for running the training loop
    and calculating the final weights for further predictions.
    
    Depedending on the user input, it either applies Gradient Descent
    or Normal Equation to calculate the weights.
    """

    def fit(self) -> None:
        if self.__method == 'gradient':
            self.__gradient_descent()
        elif self.__method == 'normal':
            self.__normal_equation()
        else:
            raise Exception('Incorrect training method given')

    """
    Function responsible for running the Gradient Descent
    algorithm to calculate the final weights for further
    predictions.
    """

    def __gradient_descent(self) -> None:
        for i in range(self.__iterations):
            # Calculating the initial predictions
            # According to the present weights.
            y_pred = self.predict(self.__X)

            # Calculating the error incurred
            loss = y_pred - self.__Y

            # Calculating the derivative of error
            gradient = self.__X.T.dot(loss) / self.__num_features

            # Updating the weights matrix by
            # subtracting the gradient from it.
            self.__weights = self.__weights - (gradient * self.__learning_rate)

    """
    Function responsible for direct calculation of
    weights by using the normal equation.
    """

    def __normal_equation(self) -> None:
        self.__weights = np.linalg.inv(self.__X.T.dot(self.__X)).dot(self.__X.T.dot(self.__Y))

    """
    Function responsible for predicting value
    for the given values of X by calculating a
    weighted sum of X.
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.__weights)
