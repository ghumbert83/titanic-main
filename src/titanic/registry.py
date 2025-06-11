"""
Save and load models, preprocessors
"""

import pickle, os

def save_model(model, path: str, filename: str) -> None:
    """
    Save a model to the specified path.
    Args:
        model (object): The model to be saved.
        path (str): The directory where the model will be saved.
        filename (str): The name of the file to save the model as.
    Returns:
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename), "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> object:
    """
    Load a model from the specified path.

    Args:
        path (str): The file path from which the model will be loaded.

    Returns:
        The loaded model.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    if not model:
        raise ValueError(f"Model at {filepath} is empty or could not be loaded.")
    return model



if __name__ == "__main__":
    print(pickle.dumps({"a": 1, "b": 2, "c": 3}))
