"""
Train the Titanic model.
"""

from titanic.data import clean_data, prepare_data, load_data
from sklearn.linear_model import LogisticRegression


def train_model() -> object:
    """
    Train the Titanic model using logistic regression.
    Returns:
        object: The trained logistic regression model.
    """
    train_df, _ = load_data()
    train_df_cleaned = clean_data(train_df)
    X_train, y_train = prepare_data(train_df_cleaned)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model) -> None:
    """
    Evaluate the trained model on the test dataset.
    Args:
        model (object): The trained model to evaluate.
    """
    _, test_df = load_data()
    test_cleaned_df = clean_data(test_df)
    X_test, y_test = prepare_data(test_cleaned_df)
    print("Model evaluation completed. Accuracy:", model.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV
def optimize_model(model) -> object:
    """
    Optimize the model using GridSearchCV.
    Args:
        model (object): The model to optimize.
    Returns:
        object: The optimized model.
    """

    train_df, _ = load_data()
    train_df_cleaned = clean_data(train_df)
    X_train, y_train = prepare_data(train_df_cleaned)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_, "Best score:", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    return best_model


if __name__ == "__main__":
    model = train_model()
    print("Model trained successfully.", model)
    evaluate_model(model)
    print("Model evaluation completed.")
    opti_model = optimize_model(model)
    print("Model optimized successfully.", opti_model)
    evaluate_model(opti_model)
    print("Optimized model evaluation completed.")
