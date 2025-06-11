"""

"""

from titanic.data import load_data,clean_data,prepare_data
from titanic.registry import save_model
from titanic.train import train_model, evaluate_model, optimize_model
def main():
    # Load the data
    train_df, test_df = load_data()

    # Clean the data
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # Prepare the data
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)

    # Train the model
    model = train_model()

    # Evaluate the model
    evaluate_model(model)

    # Optimize the model
    optimized_model = optimize_model(model)

    # Save the model
    save_model(optimized_model, "models", "model.pkl")
  
main()