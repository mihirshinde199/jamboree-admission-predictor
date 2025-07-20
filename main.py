import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.preprocessing as prep
import src.modeling as mdl
import src.evaluation as eval

# Load data
df = prep.load_data('data/jamboree_admission.csv')
prep.check_missing_duplicates(df)

# Feature / Target Split
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Check VIF
vif = eval.vif_scores(X_train_scaled_df)
print("VIF Scores:\n", vif)

# Train statsmodel regression
model = mdl.train_linear_regression(X_train_scaled_df, y_train)
print(model.summary())

# Save summary
with open("outputs/model_summary.txt", "w") as f:
    f.write(str(model.summary()))

# Evaluate
eval.evaluate_metrics(model, X_test_scaled_df, y_test)
eval.plot_residuals(model, X_test_scaled_df, y_test)
