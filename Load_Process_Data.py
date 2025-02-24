import pandas as pd

def load_data(filename1, filename2):
    x = pd.read_csv(filename1)
    y = pd.read_csv(filename2)['PAM50'] 
    return x, y