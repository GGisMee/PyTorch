import pandas as pd
from sys import path
import numpy as np

class transposeAI:
    def __init__(self, input_file_name: str, output_file_name:str, col1_is_rownames: bool = False, dir = False, append_start_col_header: bool = True) -> None:
        '''Transposes a csv file with string values into one with integer values ex column1 = ['Male', 'Female', ..., 'Female'] into [[1,0,...,0],[0,1,...,1]]
        
        Input:
            input_file_name: str = the csv filename to change, add .csv
            output_file_name: str = the new csv filename to output it to, add .csv
            col1_is_rownames: bool = if column nr 1 is a rowname, then it will not be processed like the other data, but remain unprocessed
            dir: bool/str = if false: path[0], else: dir which the input and output files are in
            append_start_col_name: bool = If the start column header should be appender to the new header. ex start_header = 'Gender', new_header = ['Male', 'Female'], final_header = ['Gender Male', 'Gender Female'] if true
        '''
        input_file_name, output_file_name = self.preprocess_data(input_file_name, output_file_name, dir)
        column_names, row_names, data, row_name_column = self.get_data(input_file_name, col1_is_rownames)
        data, classes = self.disect_data(data, column_names, append_start_col_header)

        dataframe =  self.turn_table(data, classes, row_names, col1_is_rownames, row_name_column)

        self.create_csv(dataframe, output_file_name)

    #* input csv
    def preprocess_data(self, input_file_name, output_file_name, dir):
        if not dir:
            dir = path[0]
        input_file_name = f'{dir}/{input_file_name}'
        output_file_name = f'{dir}/{output_file_name}'
        return input_file_name, output_file_name

    def get_data(self,input_file_name,col1_is_rownames):
        dataframe = pd.read_csv(input_file_name)
        column_names = dataframe.columns
        data = dataframe.to_numpy()

        row_names = None
        if col1_is_rownames:
            row_names = data[:,0]
            data = data[:,1:]
            row_name_column = column_names[0]
            column_names = column_names[1:]
        return column_names, row_names, data, row_name_column

    def change_data(self,column):
        # Removes any empty cells
        column[[isinstance(x, float) for x in column]] = 'Other'

        # Checks for all the column items
        classes = np.unique(column)

        # Creates the frame for the new data
        new_data = np.zeros((len(column), len(classes)))

        for i, class_ in enumerate(classes):
            column[column == class_] = i
        for i,row in enumerate((new_data)):
            new_data[i][column[i]] = 1
        return new_data, classes



    def disect_data(self,columns, column_names, append_start_col_header: bool):
        new_data = np.zeros((len(columns), 1))
        new_data_stacked = [new_data]
        classes = []
        for i, column in enumerate(np.transpose(columns)):
            transformed_data, classes_column = self.change_data(column)
            #print(transformed_data[0], classes_column)
            #print(len(transformed_data[0]), len(classes_column))
            print(column_names[i], classes_column)
            if append_start_col_header:
                classes_column = [f'{column_names[i]} {el}' for el in classes_column]
            classes.append(classes_column)
            new_data_stacked.append(transformed_data)

        # Flattens the array, later for column headers
        classes = np.concatenate(classes)

        data = np.hstack(new_data_stacked)[:,1:]
        #print(classes, data[0])
        #print(len(classes), len(data[0]))
        return data, classes

    def turn_table(self, data, classes, row_names, col1_is_rownames, row_name_column):
        if col1_is_rownames:
            #print(row_names.shape, data.shape)
            row_names = row_names[:, np.newaxis]
            data = np.hstack((row_names,data))

            classes = np.hstack((row_name_column, classes))
        dataframe = (pd.DataFrame(data, columns = classes))
        print(dataframe)
        return dataframe

    def create_csv(self, dataframe, output_file_name):
        print(dataframe)
        dataframe.to_csv(output_file_name, index = False)

class fuseCSV:
    def __init__(self, left_file: str, right_file:str, new_file:str, dir: bool = False, del_row_1_right: bool = False):
        '''Fuses two csv files to combine their data
        
        variables:
            left_file: str =  The filename which data should be left
            right_file: str = The filename which data should be right
            new_file = The filename where the new data should be placed
            dir: str = The dir which the files are in
            del_row_1_right: bool =  If the first column in the right_file should be deleted, since it might be row_names'''
        left_data = self.get_data(left_file, dir, False)
        right_data = self.get_data(right_file, dir, del_row_1_right)
        new_data = self.fuse_data(left_data, right_data)
        self.create_csv(new_data, new_file, dir)

    def create_csv(self, new_data, new_file, dir):
        if not dir:
            dir = path[0]
        column_names = new_data[0]
        new_data = new_data[1:]
        pd.DataFrame(new_data, columns = column_names).to_csv(f'{dir}/{new_file}', index = False)

    def fuse_data(self,left_data, right_data):
        new_data = np.hstack((left_data, right_data))
        return new_data

    def get_data(self, file,dir, del_row_1: bool):
        if not dir:
            dir = path[0]
        dataframe = pd.read_csv(f'{dir}/{file}')
        # making the column headers a part of the data
        data = pd.concat([dataframe.columns.to_frame().T, dataframe]).to_numpy()
        if del_row_1:
            data = data[:,1:]
        return data
    
        

