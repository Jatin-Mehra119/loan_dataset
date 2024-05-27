import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Function to compute annual income
def compute_annual_income(df):
    df['annual.inc'] = np.exp(df['log.annual.inc'])
    return df

# Function to categorize income
def categorize_income(df):
    def income_cat(x):
        if x <= df['annual.inc'].quantile(0.35):
            return 'lower'
        elif x <= df['annual.inc'].quantile(0.50):
            return 'lower-middle'
        elif x <= df['annual.inc'].quantile(0.75):
            return 'middle'
        elif x <= df['annual.inc'].quantile(0.85):
            return 'upper-middle'
        else:
            return 'high'
    
    df['income_cat'] = df['annual.inc'].apply(income_cat)
    return df

# Function to categorize credit scores
def categorize_credit_score(df):
    def credit_score_cat(x):
        if x >= 800:
            return 'Excellent'
        elif x >= 740:
            return 'Very good'
        elif x >= 670:
            return 'Good'
        elif x >= 580:
            return 'Fair'
        else:
            return 'Poor'
    
    df['credit_score_cat'] = df['fico'].apply(credit_score_cat)
    return df

# Function to convert columns to boolean
def convert_to_boolean(df):
    df['not.fully.paid'] = df['not.fully.paid'].astype(bool)
    return df

class ComputeAnnualIncome(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return compute_annual_income(X.copy())

class CategorizeIncome(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return categorize_income(X.copy())

class CategorizeCreditScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return categorize_credit_score(X.copy())

class ConvertToBoolean(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return convert_to_boolean(X.copy())
