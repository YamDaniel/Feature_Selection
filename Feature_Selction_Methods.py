# Step 2: Feature Selection Methods

def variance_threshold_selection(X, threshold=0.01):
    selector = VarianceThreshold(threshold)
    return selector.fit_transform(X)

def select_k_best(X, y, k=50):
    selector = SelectKBest(mutual_info_classif, k=k)
    return selector.fit_transform(X, y)

# Step 3: Model Training & Evaluation

def train_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Step 4: Run Pipeline

def run_pipeline():
    X, y = load_data()
    X_var = variance_threshold_selection(X)
    X_kbest = select_k_best(X_var, y)
    accuracy = train_evaluate(X_kbest, y)
    print(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_pipeline()