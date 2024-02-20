# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
#import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV





def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    y_train = train_df['price_CHF'][train_df['price_CHF'].notnull()]
    X_train = train_df[train_df['price_CHF'].notnull()].drop(['price_CHF'],axis=1)
    X_test = test_df

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    dummies_train = pd.get_dummies(X_train['season'])
    X_train = pd.concat([X_train, dummies_train], axis = 1).drop(['season'], axis = 1)
    
    dummies_test = pd.get_dummies(X_test['season'])
    X_test = pd.concat([X_test, dummies_test], axis = 1).drop(['season'], axis = 1)
    
    
    knn = KNNImputer(weights= 'distance')
    X_train = knn.fit_transform(X_train)
    X_test = knn.fit_transform(X_test)


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test



def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    
    param_grid = [{"kernel": [WhiteKernel(noise_level= noise) + RBF(length_scale = ls) 
                              for noise in np.logspace(-2, 0.5, 10) 
                              for ls in np.logspace(-1, 0.5, 10)]},
                  {"kernel": [Matern(length_scale = ls, nu = nu) + RBF(length_scale = ls) 
                              for nu in [0.5, 1.5, 2.5] 
                              for ls in np.logspace(-2, 0.5, 10)]},
                  {"kernel": [RationalQuadratic(alpha = alpha, length_scale=ls) + RBF(length_scale = ls) 
                              for alpha in np.logspace(-2, 0.5, 10) 
                              for ls in np.logspace(-1, 0.5, 10)]},
                  {"kernel": [WhiteKernel(noise_level= noise) + Matern(length_scale = ls, nu = nu)
                              for noise in np.logspace(-2, 0.5, 10) 
                              for nu in [0.5, 1.5, 2.5] 
                              for ls in np.logspace(-1, 0.5, 10)]},
                  {"kernel": [WhiteKernel(noise_level= noise) + RationalQuadratic(alpha = alpha, length_scale=ls)
                              for noise in np.logspace(-2, 0.25, 10) 
                              for alpha in np.logspace(-2, 0.5, 10) 
                              for ls in np.logspace(-1, 0, 10)]},
                  {"kernel": [Matern(length_scale = ls, nu = nu) + RationalQuadratic(alpha = alpha, length_scale=ls) 
                              for nu in [0.5, 1.5, 2.5]  
                              for alpha in np.logspace(-2, 0.5, 10) 
                              for ls in np.logspace(-1, 0.5, 10)]}]
                  
                     
    gp = GaussianProcessRegressor(normalize_y=True, random_state=40)
    rs = RandomizedSearchCV(gp,param_distributions = param_grid, n_iter = 40, cv=10, scoring="r2", n_jobs= 4, verbose = 1)
    rs.fit(X_train, y_train)
    gp_final = rs.best_estimator_
    gp_final.fit(X_train, y_train)
    y_pred = gp_final.predict(X_test)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

