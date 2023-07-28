import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
class progress_viewer():
    def __init__(self, name):
        """initialize the progress viewer"""
        self.name = name
        self.csv_path = f"{os.getcwd()}/{self.name}.csv"
        if os.path.exists(self.csv_path):
            #print("Already exists")
            self.df = pd.read_csv(self.csv_path)
        else:
            #print("Doesn't exist")
            self.df= pd.DataFrame(columns=["train_loss","train_acc","test_loss","test_acc","name"])
            self.df.to_csv(self.csv_path, index=False)
    def add_data(self,train_loss, train_acc, test_loss, test_acc, name, replace=False):
        """ Inserts data to csv
        
        Replace gives the ability to replace a name if it already exists in the df"""
        if len(self.df[self.df["name"] == name].index) != 0:
            if replace:
                index = ((self.df[self.df["name"] == name].index)[0])

                without_old_df = (self.df[self.df["name"] != name])
                without_old_df.loc[index] = [train_loss, train_acc, test_loss, test_acc, name]
                self.df = without_old_df.sort_index()
                
            else:
                print("Name already exists in Dataframe, either change name or activate replace as true")
                return
        else:
            self.df.loc[len(self.df)] = [train_loss, train_acc, test_loss, test_acc, name]
        self.df.to_csv(self.csv_path, index=False)
    def show(self, scatter: bool = True, width: int = 5):
        """if true: Scatter plot, else: line plot"""
        name = np.array(self.df["name"])
        train_loss = np.array(self.df["train_loss"])
        train_acc = np.array(self.df["train_acc"])
        test_loss = np.array(self.df["test_loss"])
        test_acc = np.array(self.df["test_acc"])

        labels = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]
        label_colors = ['red', 'green', 'blue', 'orange']
        plt.figure(figsize=(8,4))

        if scatter:
            plt.scatter(name, train_loss, c=label_colors[0], s=width*40)
            plt.scatter(name, train_acc, c=label_colors[1], s=width*40)
            plt.scatter(name, test_loss, c=label_colors[2], s=width*40)
            plt.scatter(name, test_acc, c=label_colors[3], s=width*40)
            plt.title("Plot of accuracy and loss")

            # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor='C0', markersize=10) for label in labels]
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(labels, label_colors)]
            plt.legend(handles=legend_elements, title="Labels", loc='upper left')
        else:
            plt.plot(name, train_loss, color=label_colors[0], linewidth=width, label="train loss")
            plt.plot(name, train_acc, color=label_colors[1], linewidth=width, label="train accuracy")
            plt.plot(name, test_loss, color=label_colors[2], linewidth=width, label="test loss")
            plt.plot(name, test_acc, color=label_colors[3], linewidth=width, label = "test accuracy")
            plt.legend()
        plt.show()
    def show_apart(self, index, scatter: bool = True, width: int = 5):
        """choose by index from: train_loss, train_accuracy, test_loss, test_accuracy
        
        if true: Scatter plot, else: line plot"""
        name = np.array(self.df["name"])
        chosen = ["train_loss", "train_acc", "test_loss", "test_acc"][index]
        color = ['red', 'green', 'blue', 'orange'][index]
        data = np.array(self.df[chosen])
        plt.figure(figsize=(8,4))
        if scatter:
            plt.scatter(name, data, c=color, s=width*40)
            plt.title(f"Plot of {chosen}")
        plt.show()
    def clear(self):
        """clears all the rows of data"""
        self.df = pd.DataFrame(columns=self.df.columns)
        self.df.to_csv(self.csv_path, index=False)
    def delete(self):
        os.remove(self.csv_path)
        
def all_csv():
    all_files = os.listdir(os.getcwd())
    return [file for file in all_files if file.endswith(".csv")]
d = progress_viewer("viewer_data")
d.add_data(19,15,27,49, "1")