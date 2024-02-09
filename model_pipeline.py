from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

param_grid_dt = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6]
}



def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

def train_and_evaluate(X, y):
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), param_grid_lr),
        "Random Forest": (RandomForestClassifier(random_state=42), param_grid_rf),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), param_grid_dt),
        "XGBClassifier": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb)
    }

    results_summary = ""  # Initialize an empty string to append results to
    X_train_resampled, X_test, y_train_resampled, y_test = prepare_data(X, y)

    for name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        results_summary += f"\n{name} Results:\n"
        results_summary += f"Best Parameters: {grid_search.best_params_}\n"
        results_summary += f"{classification_report(y_test, y_pred)}"
        results_summary += f"ROC-AUC Score: {roc_auc_score(y_test, y_pred)}\n"

    print(results_summary)  # Add this line in train_and_evaluate before the return statement
    return results_summary



def plot_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Diseases and Factors')
    plt.show()
