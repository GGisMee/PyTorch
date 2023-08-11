import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt
import os
from sys import path
from typing import List, Callable, Union
from torchvision import datasets
from tqdm.auto import tqdm


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
    def load(self,  name: str = "dataframe", path: str = path[0], create:bool=False):
        """load df from path/name"""
        if os.path.exists(f"{path}/{name}.csv"):
            self.df = pd.read_csv(f"{path}/{name}.csv")
            return
        if create:
            self.df.to_csv(f"{path}/{name}.csv", index=False)
            return
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
        super().__init__(column_names = ["test_loss","test_acc","time", "device", "name"])
    def add(self, test_loss, test_acc,time,device, name, replace=False):
        """just a packager for parent add"""
        if name not in self.df["name"].values:
            super().add([test_loss, test_acc, time,device, name])
        elif replace:
            self.df.loc[self.df["name"]==name, ["test_loss", "test_acc", "time","device"]] = [test_loss, test_acc, time, device]
            print(self.df)
    def add_dict(self,model_results:dict, time,device, name, replace=False):
        """A packager for add, model_results from Model_operations.eval_model()"""
        acc,loss = list(model_results.values())[1:]
        self.add(loss, acc,time,device, name, replace)
    def view(self):
        values = self.df.to_numpy().T
        names = values[-1]
        device = values[-2]
        values = values[:-2]
        colors = ["#8ecae6","#219ebc","#ffb703","#fb8500","#ff006e","#8338ec","#3a86ff","#3a0ca3","#0081a7", "#00afb9", "#fdfcdc", "#fed9b7"]
        # %timeit get_color_2(names)
        plt.figure(figsize=(12,6))

        for i,el in enumerate(["Accuracy","Loss","Time"]):
            print(names, values[i])
            plt.subplot(1, 3, i+1)
            if el != "Time":
                plt.bar(names, values[i], color=colors[i*3:3*(i+1)])
            else:
                plt.bar(names, values[i], color=colors[i*3:3*(i+1)], label=device)
                plt.legend()
            plt.title(el)

        plt.show()
    def save(self,name: str = "dataframe_diff_viewer", path = path[0], overwrite:bool=False):
        print(name, path)
        super().save(name, path, overwrite)
    def load(self,name: str = "dataframe_diff_viewer", path = path[0],create:bool=False):
        super().load(name, path, create)

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
class save_load:
    """Save and load models
    
    funcs:
        save_state_dict(): Saves a models statedict
        load_state_dict(): Loads a models statedict
        save_model():
        load_model():
        """
    def save_state_dict(model, name:str = "model", path = path[0]):
        """saves a models statedict

        args:
            model (model_class): the model which:s statedict will be saved
            name (str): the name of the file where the models state dict will be stored
            path (str): the directory where the file will be stored in
        """
        model_full_path = f"{path}/{name}.pth" # pth, pth eller pt for pytorch
        pt.save(obj=model.state_dict(), f=model_full_path)
    def load_state_dict(loaded_model, name:str = "model", path = path[0]) -> pt.nn.Module:
        """loads a models statedict

        args:
            loaded_model (model): the selected model which the state dict will be loaded on
            name (str): the name of the file where the models state dict is stored
            path (str): the directory where the file is stored in

        returns:
            load_model (model_class): the model"""
        loaded_model.load_state_dict(pt.load(f=f"{path}/{name}.pth")) # f is a file like object, that can be stringified
        return loaded_model
    def save_model(model, name:str = "model", path = path[0]):
        """Save a whole model
        
        args:
            model: The model which is saved
            name: The name of the file where the model gets saved
            path: The path where the file gets stored"""
        pt.save(obj=model, f=f"{path}/{name}.pth")
    def load_model(name:str = "model", path = path[0]):
        """Load a whole model
        
        args:
            name: The name of the file where the model got saved
            path: The path where the file exists"""
        pt.load(f=f"{path}/{name}.pth")
#* Does Model operations
class Model_operations:
    """Does operations on the chosen model
    
    functions:
        eval_model() - evaluates the models performence.
        train_step() - Performs a testing loop step on model going over data_loader.
        test_step() - Performs a testing loop step on model going over data_loader.
        make_predictions() - Predicts the labels on a Tensor of pictures as Tensors.

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
            diff_viewer: if you want to directly append your results to a diff_viewer then add a diff viewer

        returns:
            if dict:
            {model_name: value,
            model_loss: value,
            model_acc: value}

            if list:
            [model_loss, model_acc]
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

    def make_predictions(model:pt.nn.Module, sample:list, device:pt.device) -> pt.Tensor:
        """Makes predictions on inputed sample, which should be a list with image tensors inside examplewise [pt.tensor(...),pt.tensor(...),pt.tensor(...),...]
        
        args:
            model: The model, which the predictions are made from
            sample: The sample which the model predicts on
            device: The device the predictions are made on

        returns:
            A tensor of the targets/labels which are outputed
            """
        pred_probs = []
        model.to(device)
        model.eval()
        with pt.inference_mode():
            for picture_tensor in sample:
                # Prepare picture_tensor
                picture_tensor = pt.unsqueeze(picture_tensor, dim=0).to(device)

                # forward pass
                y_logits = model(picture_tensor)

                # Get prediction probability (logit -> prediction probability)
                pred_prob = pt.softmax(y_logits.squeeze(), dim=0)

                # Get pred_prob to cpu, to work with matplotlib
                pred_probs.append(pred_prob.cpu())
            
        # Stack pred probs to turn list to tensor
        return pt.stack(pred_probs)

    def predict_all(model, dataloader, device, softmax_dim: int = 1, argmax_dim: int= 1):
        """Make prediction on trained model usin dataloader
        
        args:
            model: the model which predicts on the data
            dataloader: loader of the data
            device: the device which the calculations are done on
            softmax_dim: the dimension that softmax changes on
            argmax_dim: the dimension that argmax calculates on

        returns:
            y_preds: list = predictions"""
        y_preds = []
        model.eval()
        with pt.inference_mode():
            for X, y in tqdm(dataloader, desc="Making predictions..."):
                # Send data + targets to target device
                X = X.to(device)
                # Do the forward pass
                y_logits = model(X)
                print(y_logits.shape)
                # Logits -> pred probs
                y_pred_probs = pt.softmax(y_logits, dim=softmax_dim)

                y_pred = pt.argmax(y_pred_probs, dim=argmax_dim)
                # print(y_pred)
                # Put predictions on CPU for evaluation for matplotlib
                y_preds.append(y_pred)

        # Concatinate list of predictions to a tensor
        y_preds = pt.cat(y_preds).cpu()
        return y_preds

# Gets a sample from the dataset
def get_sample(dataset: datasets,amount:int,seed:int=None):
    """Return a sample of pictures from the dataset, usable with make_predictions

    args:
        dataset: The dataset which the data is taken from
        seed: The seed which generates randomness, None if full randomness
        amount: The amount of samples and labels
    
    returns:
        (test_labels, test_samples)"""
    import random
    random.seed(seed)
    test_samples = []
    test_labels = []
    # det den gör är att den tar 9 st samples genom värdet k från test_data
    for sample, label in random.sample(list(dataset), k=amount):
        test_samples.append(sample)
        test_labels.append(label)
    return test_labels, test_samples


class view:
    """A class which has functions that shows the pictures from the dataset
    
    funcs:
        image(): show one image
        rand_images() show many images"""

    #* view an image
    def image(img:pt.Tensor, label:str, color:plt.cm="gray"):
        """Shows the image chosen
        
        args:
            img: image tensor chosen
            label: the label describing the tensor
            color: colormap on the picture"""
        plt.imshow(img,cmap=color)
        plt.title(label)
        plt.axis(False)
        plt.show()

    # view many images
    def rand_images(dataset,seed:int=None, nrows:int=3, ncols:int=3, color: plt.cm = "gray", figsize:tuple = (9,9)):
        """Shows a grid of pictures from dataset with randomness
        
        args:
            dataset: The dataset which the data is taken from
            seed: the seed which chooses randomness
            nrows: number of rows on display
            ncols: number of columns on display
            color: colormap which colors picture
            figsize: size of figure"""
        pt.manual_seed(seed)
        fig = plt.figure(figsize=figsize)
        for i in range(1, nrows*ncols+1):
            random_idx = pt.randint(0, len(dataset), size=[1]).item()
            #print(random_idx,i)
            img, label = dataset[random_idx]
            fig.add_subplot(nrows, ncols, i)
            plt.imshow(img.squeeze(), cmap=color) #, cmap="gray"
            plt.title(dataset.classes[label])
            plt.axis(False)
    
    def images(images:List[pt.Tensor], labels, classes:list=None, nrows:int=3, ncols:int=3, color:plt.cm = "gray", figsize: tuple = (9,9)):
        """Shows images
        
        args:
            images: pictures to display
            labels: display above pictures
            classes: To turn labels to string
            nrows: number of rows
            ncols: number of columns
            color: colors the pictures
            figsize: size of figure"""
        if len(images) < nrows*ncols:
            print('Too few images')
            return 0
        if len(labels) < nrows*ncols:
            print('Too few labels')
            return 0

        if classes is not None:
            labels = [classes[el] for el in labels]
        
        nprod = nrows*ncols
        plt.figure(figsize=figsize)
        for i in range(nprod):
            plt.subplot(nrows, ncols, i+1)
            plt.imshow(images[i], cmap=color)
            plt.title(labels[i])
            plt.axis(False)
        plt.tight_layout()
        plt.show()


    def true_false(predictions:pt.Tensor, labels,samples:List[pt.Tensor], class_names:list, ncols:int, nrows:int, figsize: tuple = (9,9)):
        """Shows a sample of pictures
        
        args:
            predictions (pt.Tensor): A PyTorch tensor containing predicted labels.
            labels: A list of true labels.
            samples (List[pt.Tensor]): A list of PyTorch tensors representing images.
            class_names (list): A list of class names corresponding to label values.
            ncols (int): Number of columns in the image grid.
            nrows (int): Number of rows in the image grid.
            figsize (tuple, optional): Size of the figure (width, height). Default is (9, 9)."""
        nprod = ncols*nrows
        if len(samples) < nprod:
            print('To few samples for cols and rows')
            return 0
        plt.figure(figsize=figsize)
        fig = plt.gcf()
        fig.suptitle("Pred | True", fontsize=20)

        for i, sample in enumerate(samples):
            plt.subplot(nrows, ncols, i+1)
            plt.imshow(sample.squeeze(), cmap="gray")
            pred_label = class_names[predictions[i]]
            true_label = class_names[labels[i]]
            if pred_label == true_label:
                plt.title(f"{pred_label} | {true_label}", c="g")
            else:
                plt.title(f"{pred_label} | {true_label}", c="r")
            plt.axis(False)
        plt.tight_layout()

#* Chains functions with initial_data as the data in the first function  
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

def confusion_matrix(y_preds, targets, class_names):
    """Compute and visualize a confusion matrix
    
        args:
            y_preds (torch.Tensor): Predicted labels or probabilities.
            targets (torch.Tensor): True labels.
            class_names (list): List of class names corresponding to label values."""

    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix

    # Setup a confusion matrix with a type and a number of classes
    conf_mat = ConfusionMatrix('multiclass',num_classes=len(class_names))

    # Gets a tensor using conf_mat 
    conf_mat_tensor = conf_mat(preds=y_preds, target=targets) 
    print(f"Our confusion matrix\n: {conf_mat_tensor}")

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_mat_tensor.numpy(), # to numpy because it mpl likes np
        class_names=class_names,
        figsize=(10,7)
    )

def time_func(start:float, device:str = None):
    """Returns and prints the time between start and function call
    
    args:
        start: When the timer is start, should be a timeit.default_timer()
        device: What the timer has been used on"""
    from timeit import default_timer
    total_time = default_timer() -start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time