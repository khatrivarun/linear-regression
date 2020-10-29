import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from src.linear_regression.LinearRegressor import LinearRegressor
from src.metric.root_mean_squared_error import root_mean_squared_error

if __name__ == '__main__':
    X, Y = make_regression(n_samples=3000, n_features=1, n_targets=1, noise=70, random_state=42)

    X_train, X_test = X[:2000], X[2000:]
    Y_train, Y_test = Y[:2000], Y[2000:]

    lin_reg = LinearRegressor(x=X_train, y=Y_train, method='gradient')
    lin_reg.fit()
    Y_pred = lin_reg.predict(x=X_test)

    rmse_error = round(root_mean_squared_error(Y_test, Y_pred), 4)

    print(rmse_error)

    plt.subplot(121)
    plt.scatter(X_test, Y_test)
    plt.title('Test Dataset')

    plt.subplot(122)
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, Y_pred, 'r--')
    plt.title(f'Predicted Line (RMSE = {rmse_error})')

    plt.show()
