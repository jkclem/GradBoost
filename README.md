# GradBoost

This project was meant to be educational. XGBoost and scikit-learn's gradient boosting algorithms are strongly recommended over GradBoost. Ijust did this as an excercise.

GradBoost takes in a regression model with methods .fit(X, y) and .predict(X), a set of testing X data to be predicted, training X and y data to fit on, and returns a tuple of the predicted y-values for the training data and testing data. The boosting rounds and learning rate are adjustable. The flexibility of model choice allows custom loss functions for the weak learners.

# GradBoost
Package to perform gradient boosting for regression using sklearn style regression models.

More details on the gradient boosting are here: https://en.wikipedia.org/wiki/Gradient_boosting

## Example

The example folder in this repository applies this function to simulated data using sklearn's DecisionTreeRegressor and RidgeCV models.
