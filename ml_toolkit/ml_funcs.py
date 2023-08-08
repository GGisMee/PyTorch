import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt
import os
from sys import path
from typing import List, Callable, Union


#* data viewers
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
        super().add([train_loss, train_acc, test_loss, test_acc])
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
        super().__init__(column_names = ["test_loss","test_acc","time", "name"])
    def add(self, test_loss, test_acc,time, name, replace=False):
        """just a packager for parent add"""
        if name not in self.df["name"].values:
            super().add([test_loss, test_acc, time, name])
        elif replace:
            self.df.loc[self.df["name"]==name, ["test_loss", "test_acc", "time"]] = [test_loss, test_acc, time]
            print(self.df)
    def view(self):
        values = self.df.to_numpy().T
        names = values[-1]
        values = values[:-1]
        colors = ["#8ecae6","#219ebc","#ffb703","#fb8500","#ff006e","#8338ec","#3a86ff","#3a0ca3","#0081a7", "#00afb9", "#fdfcdc", "#fed9b7"]
        # %timeit get_color_2(names)
        plt.figure(figsize=(12,6))

        for i,el in enumerate(["Accuracy","Loss","Time"]):
            print(names, values[i])
            plt.subplot(1, 3, i+1)
            plt.bar(names, values[i], color=colors[i*3:3*(i+1)])
            plt.title(el)

        plt.show()
    def save(self,name: str = "dataframe_diff_viewer", path = path[0], overwrite:bool=False):
        print(name, path)
        super().save(name, path, overwrite)
    def load(self,name: str = "dataframe_diff_viewer", path = path[0]):
        super().load(name, path)



#* comparation

def view_results(X_train, y_train, X_test, y_test, model):
    import helper_functions
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Train")
    helper_functions.plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1,2,2)
    plt.title("Test")
    helper_functions.plot_decision_boundary(model, X_test, y_test)

#* save and load PyTorch Model

def save_model(model, name:str = "model", path = path[0]):
    """loads a model
    
    args:
        model (model_class): the model which:s statedict will be saved
        name (str): the name of the file where the models state dict will be stored
        path (str): the directory where the file will be stored in
    """
    model_full_path = f"{path}/{name}.pth" # pth, pth eller pt for pytorch
    pt.save(obj=model.state_dict(), f=model_full_path)
def load_model(model_class, name:str = "model", path = path[0]) -> pt.nn.Module:
    """loads a model
    
    args:
        model_class (class): the class of the selected model
        name (str): the name of the file where the models state dict is stored
        path (str): the directory where the file is stored in
    
    returns:
        load_model (model_class): the model"""
    loaded_model = model_class()
    loaded_model.load_state_dict(pt.load(f=f"{name}/{path}.pth")) # f is a file like object, that can be stringified
    return load_model

#* view an image
def view_image(img:pt.Tensor, label:str):
    plt.imshow(img)
    plt.title(label)
    plt.axis(False)
    plt.show()

#* Does Model operations
class Model_operations:
    """Does operations on the chosen model
    
    functions:
        eval_model() - evaluates the models performence.
        train_step() - Performs a testing loop step on model going over data_loader.
        test_step() - Performs a testing loop step on model going over data_loader.

    """
    # evaluate model
    def eval_model(model:pt.nn.Module, 
                   data_loader: pt.utils.data.DataLoader, 
                   loss_fn: pt.nn.Module,
                   accuracy_fn,
                   device:str):
        """Evaluates the models performence and returns a dictionary containing the results of model predicting on data_loader

        args:
            model: the chosen model
            data_loader: the data_loader from which the data is loaded from
            loss_fn: function which calculates the loss
            accuracy_fn: function which calculates the accuracy

        returns:
            {model_name: value,
            model_loss: value,
            model_acc: value}
            """
        loss, acc = 0,0
        model.eval()
        with pt.inference_mode():
            for X,y in data_loader:
                X,y = X.to(device), y.to(device)
                y_pred_logits = model(X)

                # Accumulate the loss and acc values per batch
                loss += loss_fn(y_pred_logits, y)
                acc += accuracy_fn(y, y_pred_logits.argmax(dim=1))

            # Get the average loss and acc per batch, by deviding by total
            loss /= len(data_loader)
            acc /= len(data_loader)
        return {"model_name":model.__class__.__name__, # only works when model was created with a class
                "model_loss": loss.item(),
                "model_acc": acc
                }

    # Steps through the training loop
    def train_step(model: pt.nn.Module,
               dataloader:pt.utils.data.DataLoader,
               optimizer:pt.optim.Optimizer,
               loss_fn:pt.nn.Module,
               accuracy_fn,
               device:pt.device = "cpu",
               show:bool = False) -> tuple:
        """Performs a training step with model trying to learn on data_loader

        args:
            model: the model which will be trained on
            dataloader: A generator like loader for the data
            optimizer: Optimizer which optimizes the code through gradient descend
            loss_fn: function which calculates how far from the right answer each of the predictions were
            accuracy_fn: function which calculates how meny predictions were right
            device: chosen device for the neural network to run on (cpu/gpu/tpu)
            show: if true display the loss and acc in console 

        returns:
            if show: (loss, accuracy) else None"""
        train_loss, train_acc = 0,0
        ### Training
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            # Put data to the right device
            X,y = X.to(device), y.to(device)

            # 1. Forward pass
            y_logits = model(X)

            # 2. Calculate the loss
            loss = loss_fn(y_logits, y)

            # Accumulate values
            train_loss += loss # accumulate train loss
            train_acc += accuracy_fn(y, y_logits.argmax(dim=1)) # accumulate accuracy, goes from logits -> prediction labels with argmax(dim=1)

            # optimizer zero grad, Loss backward,   Optimizer step
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Devide total train loss and acc by length of train dataloader
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        print(f"train loss: {train_loss}, train accuracy: {train_acc}") if show else None
        return (train_loss, train_acc) 

    # Steps through the test loop
    def test_step(model: pt.nn.Module,
               dataloader:pt.utils.data.DataLoader,
               loss_fn:pt.nn.Module,
               accuracy_fn,
               device:pt.device = "cpu",
               show:bool = False) -> tuple:
        """Performs a testing loop step on model going over data_loader.

        args:
            model: the model which will be trained on
            dataloader: A generator like loader for the data
            loss_fn: function which calculates how far from the right answer each of the predictions were
            accuracy_fn: function which calculates how meny predictions were right
            device: chosen device for the neural network to run on (cpu/gpu/tpu)
            show: if true display the loss and acc in console 

        returns:
            if show: (loss, accuracy) else None"""

        # Create loss and acc variables
        test_loss, test_acc = 0, 0

        # Puts the model on evaluation mode
        model.eval()

        # Turn on inference mode (Predictions mode) to look at the data  
        with pt.inference_mode():
            for X,y in dataloader:
                # Device agnostic
                X.to(device), y.to(device)

                # Forward pass
                test_logits = model(X)

                # Acumulate the loss and accuracy
                test_loss += loss_fn(test_logits, y)
                test_acc += accuracy_fn(y, test_logits.argmax(dim=1))

            # Calculate the loss (avg per batch) and accuracy
            test_loss /= len(dataloader)
            test_acc /= len(dataloader)

        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}") if show else None
        return test_loss, test_acc

    # Chains functions with initial_data as the data in the first function  
    def function_chainer(initial_data:Union[tuple, List], func_list: List[Callable]) -> any:
        """Chains functions in func_list with input as initial input and returns result

        args:
            initial_data (tuple): The initial data to be passed as arguments to the first function. 
            func_list (List[Callable]): A list of functions to be executed sequentially. 
            
        Order: [a,b,c] -> c(b(a(initial_data)))

        returns:
            any: The final output after applying all the functions."""
        output = func_list[0](*initial_data)
        for func in func_list[1:]:
            output = func(output)
        return output