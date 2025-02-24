from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def feature_selection(x, y):

    # Initialize random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(x, y)

    # Select important features
    selector = SelectFromModel(rf_model, max_features=1000, threshold="mean")
    x_new = selector.transform(x)
    print(x_new.shape)  # Shape of data after feature selection
    return x_new

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Initialize Logistic Regression with L1 regularization (multiclass)
lasso = LogisticRegressionCV(penalty='l1', solver='saga', multi_class='ovr', max_iter=10000)

# Fit the model
lasso.fit(x_scaled, y_encoded)

# Get the selected features
selected_features = lasso.coef_ != 0  # Features with non-zero coefficients
x_new = x_scaled[:, selected_features]

print(x_new.shape)  # Shape after feature selection