# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, Lars
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, LeaveOneOut
import seaborn as sns



def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((X.shape[0], 21))
    X_transformed[:,0:5] = X
    X_transformed[:,5:10] = np.power(X, 2)
    X_transformed[:,10:15] = np.exp(X)
    X_transformed[:,15:20] = np.cos(X)
    X_transformed[:,20] = 1
    assert X_transformed.shape == (700, 21)
    return X_transformed


def Hyper_tuning(X, y):
    
    # Transform X
    X_transformed = transform_data(X)
    
    # Initialize Leave-one-out cross validation
    leave_out = LeaveOneOut()
    
    models= [Ridge(fit_intercept=False, random_state = 2),
             Lasso(fit_intercept=False, random_state = 2),
             ElasticNet(fit_intercept=False, random_state = 2),
             SGDRegressor(fit_intercept=False, random_state=2)
                   ]
    
    # Define set of possible hyperparameter values
    grid = {"alpha": [0.01, 0.1, 1, 2,  5, 10, 20, 50]}

    # Apply Grid search and add for each model the best score and the respective parameter to the list
    gs_bestscore = []
    gs_bestpara = []
    for model in models:
        gs = GridSearchCV(model, param_grid = grid, cv=leave_out, scoring="neg_root_mean_squared_error", n_jobs= 4, verbose = 1)
        gs.fit(X_transformed, y)
        gs_bestscore.append(gs.best_score_)
        gs_bestpara.append(gs.best_estimator_)
    
    # Create dataframe to compare scores of each model
    gs_res = pd.DataFrame({"GS": gs_bestscore,"CrossValerrors": gs_bestpara,
                           "Algorithm":["Ridge", "Lasso", "ElasticNet", "SGD"]})
    print(gs_res)
    
    # Return best model with the respective optimal hyperparameter
    return gs_bestpara[gs_bestscore.index(max(gs_bestscore))]



def fit(X, y):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    X_transformed = transform_data(X)
    LinMod = Hyper_tuning(X, y).fit(X_transformed, y)
    w = LinMod.coef_
    assert w.shape == (21,)
    return w



if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
