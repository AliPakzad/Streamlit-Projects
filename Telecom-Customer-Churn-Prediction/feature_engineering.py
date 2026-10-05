# feature_engineering.py
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create engineered features for the churn dataset.
    This class MUST be in a separate module so it can be imported by both
    the notebook (for training) and the Streamlit app (for deployment).
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Average Monthly Spend
        X['AvgMonthlySpend'] = X['TotalCharges'] / (X['tenure'] + 1)
        
        # 2. Total Services Count
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies']
        X['TotalServices'] = X[service_cols].sum(axis=1)
        
        # 3. Is New Customer
        X['IsNewCustomer'] = (X['tenure'] <= 12).astype(int)
        
        # 4. Family Commitment
        X['FamilyCommitment'] = X['Partner'] + X['Dependents']
        
        # 5. Contract Risk
        contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
        X['ContractRisk'] = X['Contract'].map(contract_risk)
        
        # 6. Contract-Charge Interaction
        X['ContractChargeInteraction'] = X['ContractRisk'] * X['MonthlyCharges']
        
        # 7. Drop redundant columns
        cols_to_drop = ['TotalCharges', 'Partner', 'Dependents', 'Contract']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        
        return X