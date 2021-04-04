import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

class PolynomialRegressionCreator :

    def __init__(self, file) :
        self.dataset = pd.read_csv("Data.csv")
        self.featureMatrix = self.dataset.iloc[ : , : -1]
        self.resultMatrix = self.dataset.iloc[ : , -1]
        self.preProcessing()

    def preProcessing(self) :
        self.featureMatrix_train, self.featureMatrix_test, self.resultMatrix_train, self.resultMatrix_test = train_test_split(self.featureMatrix, self.resultMatrix, test_size = 0.2, random_state = 0)
        poly_features = PolynomialFeatures(degree=5)
        self.featureMatrix_train_poly = poly_features.fit_transform(self.featureMatrix_train)
        self.featureMatrix_test_poly = poly_features.transform(self.featureMatrix_test)
        self.makeModel()

    def makeModel(self) :
        self.regressor = LinearRegression()
        self.regressor.fit(self.featureMatrix_train_poly, self.resultMatrix_train)
        self.predictor()
        
    def predictor(self) :
        self.resultMatrix_pred = self.regressor.predict(self.featureMatrix_test_poly)

    def accuracyCounter(self) :
        return r2_score(self.resultMatrix_test, self.resultMatrix_pred)
