# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:41:01 2020

@author: jkcle
"""

import typing
import numpy as np

def GradBoost(model,
              X_test: np.array,                  # testing independent variables
              X_train: np.array,                 # training independent variables
              y_train: np.array,                 # training dependent variable
              boosting_rounds: int = 100,        # number of boosting rounds
              learning_rate: float = 0.1,        # learning rate with default of 0.1
              verbose: bool = True) -> np.array: # if True, shows a tqdm progress bar
    '''
    Takes in a model and performs gradient boosting using that model. This allows for 
    almost any scikit-learn model to be used.
    '''
    import numpy as np
    
    # make a first guess of our training target variable using the mean
    y_hat_train = np.repeat(np.mean(y_train), len(y_train))
    # initialize the out of sample prediction with the mean of the training target variable
    y_hat_train_test = np.repeat(np.mean(y_train), len(X_test))
    # calculate the residuals from the training data using the first guess
    pseudo_resids = y_train - y_hat_train
    
    # performs gradient boosting with a tqdm progress bar
    if verbose:
        from tqdm import tqdm_notebook as tqdm
        # iterates through the boosting round
        for _ in tqdm(range(0, boosting_rounds)):
            # fit the model to the pseudo residuals
            model = model.fit(X_train, pseudo_resids)   
            # increment the predicted training y with the pseudo residual * learning rate
            y_hat_train += learning_rate * model.predict(X_train)       
            # increment the predicted test y as well
            y_hat_train_test += learning_rate * model.predict(X_test)
            # calculate the pseudo resids for next round
            pseudo_resids = y_train - y_hat_train 
    # performs gradient boosting without a progress bar        
    else:
        # iterates through the boosting round
        for _ in range(0, boosting_rounds):
            # fit the model to the pseudo residuals
            model = model.fit(X_train, pseudo_resids)   
            # increment the predicted training y with the pseudo residual * learning rate
            y_hat_train += learning_rate * model.predict(X_train)       
            # increment the predicted test y as well
            y_hat_train_test += learning_rate * model.predict(X_test)
            # calculate the pseudo resids for next round
            pseudo_resids = y_train - y_hat_train  

    # return a tuple of the predicted training y and the predicted test y
    return y_hat_train, y_hat_train_test
