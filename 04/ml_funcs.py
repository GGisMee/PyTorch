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
                   device:str):
        """Evaluates the models performence and returns a dictionary containing the results of model predicting on data_loader

        args:
            model: the chosen model
            data_loader: the data_loader from which the data is loaded from
            loss_fn: function which calculates the loss
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
                
                y_preds = pt.argmax(pt.softmax(y_pred_logits, dim=1), dim=1)
                acc += pt.eq(y_preds, y).sum().item()/len(y_preds)

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
               loss_fn: pt.nn.Module, 
               optimizer:pt.optim.Optimizer, 
               device:pt.device, 
               show:bool=False):
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
            (loss, accuracy)"""
        # Put model in training mode
        model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0,0

        # Loop through data loader batches
        for X,y in dataloader:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            train_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_preds = pt.argmax(pt.softmax(y_logits, dim=1), dim=1) # Softmax is actually unnecessary, but can be useful for visualization and also to give completeness
            train_acc += pt.eq(y_preds, y).sum().item()/len(y_preds)
        train_loss /= len(dataloader)
        train_acc  /= len(dataloader)

        if show:
            print(f'Train loss: {train_loss} | Train acc: {train_acc}')
        return train_loss, train_acc

    # Steps through the testing loop
    def test_step(model: pt.nn.Module, 
              dataloader:pt.utils.data.DataLoader, 
              loss_fn: pt.nn.Module, 
              device:pt.device, 
              show:bool=False):
        """Performs a testing loop step on model going over data_loader.

        args:
            model: the model which will be trained on
            dataloader: A generator like loader for the data
            loss_fn: function which calculates how far from the right answer each of the predictions were
            accuracy_fn: function which calculates how meny predictions were right
            device: chosen device for the neural network to run on (cpu/gpu/tpu)
            show: if true display the loss and acc in console 

        returns:
            (loss, accuracy)"""
        test_acc, test_loss = 0,0

        model.eval()
        with pt.inference_mode():
            for X,y in dataloader:
                X,y = X.to(device), y.to(device)
                y_logits = model(X)
                loss = loss_fn(y_logits, y)
                test_loss+=loss.item()

                y_preds = pt.argmax(pt.softmax(y_logits, dim=1), dim=1)
                test_acc += pt.eq(y_preds, y).sum().item()/len(y_preds)
        test_loss /= len(dataloader)
        test_acc  /= len(dataloader)
        if show:
            print(f'Test loss: {test_loss} | Test acc: {test_acc}')


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
        rand_images() show many images
        images(): Similar to image, but with many images
        true_false(): Display which images that it got right and which it got wrong
        plot_transformed_images(): Visualizes transformed and untransformed images"""

    #* view an image
    def image(img:pt.Tensor, label:str, color:plt.cm="gray"):
        plt.imshow(img,cmap=color)
        plt.title(label)
        plt.axis(False)
        plt.show()

    # view many images
    def rand_images(dataset,seed:int=None, nrows:int=3, ncols:int=3, color: plt.cm = "gray", figsize:tuple = (9,9)):
        """Shows a grid of pictures from dataset with randomness"""
        from random import seed as random_seed
        pt.manual_seed(seed)
        random_seed(seed)
        fig = plt.figure(figsize=figsize)
        for i in range(1, nrows*ncols+1):
            random_idx = pt.randint(0, len(dataset), size=[1]).item()
            img, label = dataset[random_idx]
            fig.add_subplot(nrows, ncols, i)
            if img.shape[0] == 3: 
                img = img.permute(1,2,0)
            elif img.shape[0] == 1:
                img = img.squeeze()
            plt.imshow(img) #, cmap="gray"
            plt.title(dataset.classes[label])
            plt.axis(False)
    
    def images(images, labels, classes:list=None, nrows:int=3, ncols:int=3, color:plt.cm = "gray", figsize: tuple = (9,9)):
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


    def true_false(predictions, labels,samples, class_names, ncols, nrows, figsize: tuple = (9,9)):
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

    def plot_transformed_images(image_paths:str, transform, n:int=3, generator_seed:int=None, transform_seed:int=None):
        from PIL import Image
        """Plot difference between transformed images and normal

        args:
            image_paths (Sequence[Path()]): List of image file paths.
            transform (torchvision.transforms): Image transformation object.
            num_images (int, optional): Number of images to display. Default is 3.
            generator_seed (int, optional): Random seed for image selection. Default is None.
            generator_seed (int, optional): Random seed for transform selection. Default is None."""
        from random import sample, seed as random_seed
        random_seed(generator_seed)
        pt.manual_seed(transform_seed)
        image_paths = sample(image_paths,k=n)
        fig, ax = plt.subplots(nrows=len(image_paths),ncols=2)
        for i,img_path in enumerate(image_paths):
            with Image.open(img_path) as f:
                ax[i,0].imshow(f)
                ax[i,1].imshow(transform(f).permute(1,2,0))
                ax[i,0].axis(False)
                ax[i,0].set_title(f"Size: {f.size}")
                ax[i,1].axis(False)
                ax[i,1].set_title(f'Size: ({str(transform(f).permute(1,2,0).size())[12:-2]})')

        plt.tight_layout(w_pad=-10)

        plt.subplots_adjust(top=0.8)
        plt.suptitle('Images\nOriginal    Transformed', fontsize=20)
        plt.show()


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

from timeit import default_timer
class Timer:
    """A simple Timer
    
    funcs:
        init(): Starts the Timer
        get(): Get the Timers content and returns the values (full timer, interval list, since start list)
        interval(): Times since last interval
        since_start(): Times since initialization
        show_as_print(): Shows the output in a formated way"""
    def __init__(self):
        """Starts the timer"""
        self.starttimer = default_timer()
        self.since_last = [['start', self.starttimer]]
        self.since_start_list = [['start', self.starttimer]]
        self.stop_value = None
    def interval(self, name:str = 'Interval N'):
        """creates instance with time since last instance"""
        interval_value = default_timer()
        if name == 'Interval N':
            name = f'Interval {len(self.since_last)}'
        self.since_last.append([name, interval_value])
    def since_start(self, name:str = 'Timer N'):
        """creates instance with time since start"""
        partial_stop_v = default_timer()
        if name == 'Timer N':
            name = f'Timer {len(self.since_start_list)}'
        self.since_start_list.append([name, partial_stop_v-self.starttimer])

    def get(self) -> tuple[int, list, list]:
        """returns (full timer, interval list, since start list)"""
        self.stop_value = default_timer()-self.starttimer
        return (self.stop_value, self.since_last, self.since_start_list)
    
    def show_as_print(self, start_interval_since_decimals: tuple = (2,2,2)):
        """Prints out the values"""
        start_decimals = f'.{start_interval_since_decimals[0]}f'
        interval_decimals = f'.{start_interval_since_decimals[1]}f'
        since_decimals = f'.{start_interval_since_decimals[2]}f'
        
        text = f'''\nTimer: by GGisMee\n=================\n'''
        if not self.stop_value: # om man inte har stoppat än
            self.stop_value = default_timer()-self.starttimer   
        text+= f'''Total time:\n {self.stop_value:{start_decimals}}\n=================\n'''
        if self.since_last != [['start', self.starttimer]]:
            
            since_last_display = [[name, timeobj-self.since_last[i][1]] for i,(name, timeobj) in enumerate(self.since_last[1:])]
            since_last_display = [[name,f"{number:{interval_decimals}}"] for name,number in since_last_display]
            
            text+='Interval time: \n'
            for el in since_last_display:
                text += f'{el[0]}: {el[1]}\n'
            text+='=================\n'
        if self.since_start_list != [['start', self.starttimer]]:
            text+='Time since start: \n'
            since_start_list_display = [[name,f"{number:{since_decimals}}"] for name,number in self.since_start_list[1:]]
            for el in since_start_list_display:
                text+= f'{el[0]}: {el[1]}\n'
            text+='=================\n'
        print(text)