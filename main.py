from Multiple_Regression import MultipleRegressionCreator
from Polynomial_Regresson import PolynomialRegressionCreator
from Support_Vector_Regression import SVRRegressionCreator
from Decision_Tree_regression import DecisionTreeRegressionCreator
from Forest_Tree_Regression import RandomForestRegressionCreator

mr = MultipleRegressionCreator("Data.csv")
print("Accuracy Of Multiple Regression model" + str(mr.accuracyCounter()))
print("-----------------------------------------------------------------------")
pr = PolynomialRegressionCreator("Data.csv")
print("Accuracy Of Polynomial Regression model" + str(pr.accuracyCounter()))
print("-----------------------------------------------------------------------")
svr = SVRRegressionCreator("Data.csv")
print("Accuracy Of SVR Regression model" + str(svr.accuracyCounter()))
print("-----------------------------------------------------------------------")
dtr = DecisionTreeRegressionCreator("Data.csv")
print("Accuracy Of Decision Tree Regression model" + str(dtr.accuracyCounter()))
print("-----------------------------------------------------------------------")
rfr = RandomForestRegressionCreator("Data.csv")
print("Accuracy Of Decision Tree Regression model" + str(rfr.accuracyCounter()))