import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso

def train_linear_regression(X, y):
    """Train a linear regression model using statsmodels."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model

def train_ridge(X_train, y_train, alpha=1.0):
    return Ridge(alpha=alpha).fit(X_train, y_train)

def train_lasso(X_train, y_train, alpha=0.1):
    return Lasso(alpha=alpha).fit(X_train, y_train)
