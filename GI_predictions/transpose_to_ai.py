import pandas
from sys import path
import numpy as np

#* input csv
def get_data():
    #!file_name = input('Enter file name: ')
    file_name = "input.csv"
    dataframe = pandas.read_csv(path[0]+"/"+file_name)
    column_names = dataframe.columns
    data = dataframe.to_numpy()

    row_names = data[:,0]
    data = data[:,1:]
    return file_name, column_names, row_names, data

def change_data(column):
    print(column)

def disect_data(columns):
    for column in np.transpose(columns):
        change_data(column)



def main():
    file_name, column_names, row_names, data = get_data()
    disect_data(data)




if __name__ == '__main__':
    main()