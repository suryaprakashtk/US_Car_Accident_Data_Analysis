import pandas as pd
print("Welcome to Team 8 ECE143 Course Final Project")
# Read File
dataset_filepath = "../US_Accidents_March23.csv"
def read_csv(filename): #### NEEDS DEBUGGING - Bug prob due to big file size
    """
    Read the CSV file by converting it to a datafram to later be processed

    @param: filename -string for the name of file
    @return: dataframe object that stores the contents of the dataset
    """
    df = pd.read_csv(filename)
    return df
result = read_csv(dataset_filepath)
print(result.head())