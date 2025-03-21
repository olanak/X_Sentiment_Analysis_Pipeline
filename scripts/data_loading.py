import pandas as pd

def load_x_data(file_path):
    """loads x data downloaded from mendley

    Args:
        file_path (str): file path for the twitter dataset csv
        
    return:
        dataset : returns the loaded twitter dataset
    """
    dataset = pd.read_csv(file_path)
    return dataset