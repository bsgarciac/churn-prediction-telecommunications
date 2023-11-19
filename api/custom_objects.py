from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
      return self

    def parse_charges_float(self, x):
      if type(x)== str:
        if len(x) > 0:
          return float(x)
        else:
          return np.nan
      else:
        return x

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy.replace('No internet service', 'No', inplace=True)
        X_copy.replace('No phone service', 'No', inplace=True)
        X_copy['TotalCharges'] = X_copy['TotalCharges'].apply(lambda x : self.parse_charges_float(x))
        X_copy['TotalCharges'].replace(np.nan, np.mean(X_copy['TotalCharges']), inplace=True)
        X_copy = X_copy.drop(['customerID'],axis=1)
        X_copy['SeniorCitizen'] = X_copy['SeniorCitizen'].astype(str)
        return X_copy
    

class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator = None):#DecisionTreeClassifier(),):
        """A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """
        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)