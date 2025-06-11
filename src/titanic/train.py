"""
Train the Titanic model.
"""

from titanic.data import clean_data, prepare_data, load_data
from sklearn.linear_model import LogisticRegression


def train_model() -> object:
    """
    Initiate the model and train it on the Titanic dataset.""
    """

    train_df, _ = load_data()
    train_df_cleaned = clean_data(train_df)
    X_train, y_train = prepare_data(train_df_cleaned)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model):
    _, test_df = load_data()
    test_cleaned_df = clean_data(test_df)
    X_test, y_test = prepare_data(test_cleaned_df)
    model.score(X_test, y_test)


def optimize_model(model):
    pass
