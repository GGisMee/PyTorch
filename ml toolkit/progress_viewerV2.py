import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sys import path

# data manager, original class
class data_manager:
    def __init__(self, column_names: np.array = []):
        """A progress data manager"""
        self.df = pd.DataFrame(columns=column_names)
    def add(self, row: list):
        """Adds a row to the df"""
        self.df.loc[len(self.df)] = row
    def save(self, name: str = "dataframe", path: str = path[0], overwrite=False):
        """save df as name to path"""
        if os.path.exists(f"{path}/{name}.csv") and not overwrite:
            print("File already exists, enable overwrite to save and overwrite")
            return
        self.df.to_csv(f"{path}/{name}.csv", index=False)
    def load(self,  name: str = "dataframe", path: str = path[0]):
        """load df from path/name"""
        if os.path.exists(f"{path}/{name}.csv"):
            self.df = pd.read_csv(f"{path}/{name}.csv")
        else:
            print("path/name doesn't exist")
    def clear(self):
        """clears all the rows of data"""
        self.df = pd.DataFrame(columns=self.df.columns)
    def all_csv():
        all_files = os.listdir(path[0])
        return [file for file in all_files if file.endswith(".csv")]


class progress_viewer(data_manager):
    def __init__(self):
        super().__init__(column_names = ["train_loss","train_acc","test_loss","test_acc"])
    def add(self,train_loss, train_acc, test_loss, test_acc):
        """just a packager for parant add"""
        super().add([train_loss*100, train_acc, test_loss*100, test_acc])
    def view(self):
        for column_name in self.df.columns:
            column_value = self.df[column_name].to_numpy()
            plt.plot(column_value, label=column_name)
        plt.legend()
        plt.show()
    def save(self,name: str = "dataframe_prog_viewer", path = path[0]):
        super().save(name, path)
    def load(self,name: str = "dataframe_prog_viewer", path = path[0]):
        super().load(name, path)

class difference_viewer(data_manager):
    def __init__(self):
        super().__init__(column_names = ["train_loss","train_acc","test_loss","test_acc", "name"])
    def add(self, train_loss, train_acc, test_loss, test_acc, name, replace=False):
        """just a packager for parant add"""
        if name not in self.df["name"].values:
            super().add([train_loss*100, train_acc, test_loss*100, test_acc, name])
        elif replace:
            self.df.loc[self.df["name"]==name, ["train_loss", "train_acc", "test_loss", "test_acc"]] = [train_loss, train_acc, test_loss, test_acc]
            print(self.df)
        else:
            print("Row with name already exists, please change name or enable replace")    
    def show(self):
        labels = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]
        label_colors = ['red', 'green', 'blue', 'orange']
        plt.figure(figsize=(8,4))
        train_loss, train_acc, test_loss, test_acc, names = self.df.to_numpy().T

        plt.scatter(names, train_loss, label=labels[0], c=label_colors[0])
        plt.scatter(names, train_acc, label=labels[1], c=label_colors[1])
        plt.scatter(names, test_loss, label=labels[2], c=label_colors[2])
        plt.scatter(names, test_acc, label=labels[3], c=label_colors[3])
        plt.legend()
        plt.show()
    def save(self,name: str = "dataframe_diff_viewer", path = path[0]):
        print(name, path)
        super().save(name, path)
    def load(self,name: str = "dataframe_diff_viewer", path = path[0]):
        super().load(name, path)