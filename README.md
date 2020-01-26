# GradBoost
GradBoost takes in a regression model with methods .fit(X, y) and .predict(X), a set of testing X data to be predicted, training X and y data to fit on, and returns a tuple of the predicted y-values for the training data and testing data. The boosting rounds and learning rate are adjustable. The flexibility of model choice allows custom loss functions for the weak learners.

# GradBoost
Package to perform gradient boosting for regression using sklearn style regression models.

More details on the gradient boosting are here: https://en.wikipedia.org/wiki/Gradient_boosting

## Installation

Clone this repository, move into the directory, and install with pip:

`git clone https://github.com/jkclem/GradBoost.git`

`cd GradBoost`

`pip install .`

In your Python code you can import it as:

`from GradBoost import GradBoost`

## Example

The example folder in this repository applies this function to simulated data using sklearn's DecisionTreeRegressor and RidgeCV models.
