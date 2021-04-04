from Multiple_Regression import MultipleRegressionCreator
from Polynomial_Regresson import PolynomialRegressionCreator

mr = MultipleRegressionCreator("Data.csv")
print("Accuracy Of Multiple Regression model" + str(mr.accuracyCounter()))
print("-----------------------------------------------------------------------")
pr = PolynomialRegressionCreator("Data.csv")
print("Accuracy Of Polynomial Regression model" + str(pr.accuracyCounter()))