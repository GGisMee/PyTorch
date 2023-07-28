from pathlib import Path
import numpy as np
import pandas as pd
from sys import path
import matplotlib.pyplot as plt
class progress_viewer():
    def __init__(self, name):
        """initialize the progress viewer"""
        self.name = name
        if Path(f"{path[0]}/{self.name}.csv").is_file():
            #print("Allready exists")
            self.df = pd.read_csv(f"{path[0]}/{self.name}.csv")
        else:
            #print("Doesn't exist")
            self.df= pd.DataFrame(columns=["train_loss","train_acc","test_loss","test_acc","name"])
            self.df.to_csv(f"{path[0]}/{self.name}.csv", index=False)
    def add_data(self,train_loss, train_acc, test_loss, test_acc, name, replace=False):
        # Replace gives the ability to replace a name if it already exists in the df
        if len(self.df[self.df["name"] == name].index) != 0:
            if replace:
                index = ((self.df[self.df["name"] == name].index)[0])
                without_old_df = (self.df[self.df["name"] != name])
                without_old_df[index] = [train_loss, train_acc, test_loss, test_acc, name]
                print(without_old_df)
            else:
                print("Name already exists in Dataframe, either change name or activate replace as true")
        return
        self.df.loc[len(self.df)] = [train_loss, train_acc, test_loss, test_acc, name]
        self.df.to_csv(f"{path[0]}/{self.name}.csv", index=False)
    def show(self, scatter: bool = True, width: int = 5):
        name = np.array(self.df["name"])
        train_loss = np.array(self.df["train_loss"])
        train_acc = np.array(self.df["train_acc"])
        test_loss = np.array(self.df["test_loss"])
        test_acc = np.array(self.df["test_acc"])

        labels = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]
        label_colors = ['red', 'green', 'blue', 'orange']
        plt.figure(figsize=(12,6))

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

d = progress_viewer("data")
d.add_data(10,20, 30, 40, "frsst", 1)