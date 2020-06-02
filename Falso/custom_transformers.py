from sklearn.base import BaseEstimator, TransformerMixin


# Custom empty text transformer class for pipelines
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.field]


# Custom empty number transformer class for pipelines
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.field]]
