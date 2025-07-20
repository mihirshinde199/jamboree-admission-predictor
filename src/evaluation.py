import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def evaluate_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R² Score:", r2_score(y_test, preds))

def vif_scores(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def plot_residuals(model, X, y):
    residuals = y - model.predict(sm.add_constant(X))
    
    # Residual vs Fitted
    plt.figure(figsize=(6,4))
    sns.residplot(x=model.predict(sm.add_constant(X)), y=residuals, lowess=True)
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.savefig('outputs/residuals_plot.png')
    plt.show()
    
    # QQ Plot
    sm.qqplot(residuals, line='45')
    plt.title('QQ Plot')
    plt.savefig('outputs/qq_plot.png')
    plt.show()
