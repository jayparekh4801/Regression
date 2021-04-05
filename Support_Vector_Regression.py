import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score

class SVRRegressionCreator :

    def __init__(self, file) :
        self.dataset = pd.read_csv("Data.csv")
        self.featureMatrix = self.dataset.iloc[ : , : -1]
        self.resultMatrix = self.dataset.iloc[ : , -1]
        self.preProcessing()

    def preProcessing(self) :
        self.scaler = StandardScaler()
        self.featureMatrix_train, self.featureMatrix_test, self.resultMatrix_train, self.resultMatrix_test = train_test_split(self.featureMatrix, self.resultMatrix, test_size = 0.2, random_state = 0)
        self.featureMatrix_train_sca = self.scaler.fit_transform(self.featureMatrix_train)
        self.featureMatrix_test_sca = self.scaler.transform(self.featureMatrix_test)
        self.makeModel()

    def makeModel(self) :
        self.regressor = SVR(kernel="rbf")
        self.regressor.fit(self.featureMatrix_train_sca, self.resultMatrix_train)
        self.predictor()
        
    def predictor(self) :
        self.resultMatrix_pred = self.regressor.predict(self.featureMatrix_test_sca)

    def accuracyCounter(self) :
        return r2_score(self.resultMatrix_test, self.resultMatrix_pred)
