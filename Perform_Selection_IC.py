# -*- coding: utf-8 -*-
# creat_time: 2021/11/11 13:14


import numpy as np
import statsmodels.api as sm


def select_IC(y, F_hat, IC):
    """
    Selects the number of principal components to include in a regression
    model based on an information criterion or the adjusted R-squared
    :param y: T-vector of dependent variable observations. Note that y is a column vector
    :param F_hat: T-by-K matrix of principal component observations
    :param IC: information criterion
    :return: the selected number of factors k_star
    """
    T = y.shape[0]
    K = F_hat.shape[1]
    IC_value = np.zeros((K, 1))
    results_K = sm.OLS(y, np.concatenate([np.ones((T, 1)), F_hat], axis=1)).fit()
    p_K = K + 1;
    for k in range(K):
        X_k = np.concatenate([np.ones((T, 1)), F_hat[:, :(k+1)]], axis=1)
        result_k = sm.OLS(y, X_k).fit()
        p_k = X_k.shape[1]
        if IC == 1:
            IC_value[k] = np.log(result_k.resid.reshape(-1, 1).T @ result_k.resid.reshape(-1, 1) / T) + 2 * p_k / T
        elif IC == 2:
            IC_value[k] = np.log(result_k.resid.reshape(-1, 1).T @ result_k.resid.reshape(-1, 1) / T) + p_k * np.log(T) / T
        elif IC == 3:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                IC_value[k] = -100 * result_k.rsquared_adj

    k_star = np.nanargmin(IC_value) + 1
    return k_star   # To extract factors, one should use F_hat[, :k_star]



if __name__ == '__main__':
    y = np.arange(1, 5).reshape(-1, 1)
    F_hat = np.array([[1, 2, -100], [3, 2.20, 0], [5, 6.1, 100], [5, 4, 3]])
    k_star = select_IC(y, F_hat, IC=3)
    print(k_star)



