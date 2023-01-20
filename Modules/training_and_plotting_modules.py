import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    """Runs one training step, i.e., loops through all batches in <dataloader>
    and all training steps on each batch. The entire process is equal to the
    training part of one epoch.
    
    Returns
    -------
        train_loss: float
            Average loss per batch for this epoch.
        train_acc: float
            Average loss per batch for this epoch.
    """
    
    model.train()
    
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
      
        y_pred = model(X)
      
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred) # total num that are right / total num of samples
    
    train_loss = train_loss / len(dataloader) # average loss, i.e. loss per batch
    train_acc = train_acc / len(dataloader) # average accuracy, i.e accuracy per batch
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = device):
    """
    Runs one testing step, i.e., loops through all batches in <dataloader>
    and all testing steps on each batch. The entire process is equal to the
    testing part of one epoch.
    

    Returns
    -------
    test_loss : float
        Average loss per batch for this epoch.
    test_acc : float
        Average loss per batch for this epoch.

    """
    model.eval()
    
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
      for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        test_pred_logits = model(X)
        
        loss = loss_fn(test_pred_logits, y)
        test_loss += loss.item()
        
        test_pred_labels = test_pred_logits.argmax(dim = 1)
        test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels) # total num that are right / total num of samples

      test_loss = test_loss / len(dataloader) # average loss, i.e. loss per batch
      test_acc = test_acc / len(dataloader) # average accuracy, i.e accuracy per batch
    
    return test_loss, test_acc

def plot_loss_curves(results):
    """
    Plots training curves of a results dictionary.

    Parameters
    ----------
    results : Dict[str : List[float]]
        Contains results for one model's training and testing run.
        - The ith element in the value for each key corresponds to 
          the results for the (i+1)th epoch.
        - The keys of the dictionary must be:
            * "train_loss"
            * "train_acc"
            * "test_loss"
            * "test_acc"
        - The ith element of the value of each key (i.e. of the list
          corresponding to that key) corresponds to the results for 
          the (i+1)th epoch.

    Returns
    -------
    None.

    """
    # Get loss values of results dictionary(training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    
    epochs = range(len(results["train_loss"]))
    
    plt.figure(figsize = (15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label = "train loss")
    plt.plot(epochs, test_loss, label = "test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label = "train accuracy")
    plt.plot(epochs, test_accuracy, label = "test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
    
    return

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device = device):
    """Runs full training and testing loop for the number of epochs given.
    
    Returns
    --------
    results: Dict[str : List]
        Contains train and test loss and accuracy for each epoch.
    total_train_time: float
        Time taken in seconds by the entire training session.
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    start_time = timer()
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                           dataloader = train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer,
                                           device = device)
        test_loss, test_acc = test_step(model = model,
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device)
        print(f"Epoch: {epoch} | Train loss: {train_loss: 0.2f} | Train acc: {train_acc*100: 0.2f}% | Test loss: {test_loss: 0.2f} | Test acc: {test_acc*100: 0.2f}%")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    end_time = timer()

    total_train_time = end_time - start_time
    print(f"Training took {total_train_time: 0.1f} sec, or {total_train_time:0.1f} min")
    
    return results, total_train_time



def compare_results(results_lst, model_names_lst = None):
    """
    Plots results of all the models whose results are given in <results_lst>
    together so that they can be compared.
    
    There are 4 subplots in the plot for the following values: training loss, 
    training accuracy, testing loss, and testing accuracy. The results of the
    models on each of these are plotted on the same subplot.
    
    Parameters
    ----------
    results_lst : List[Dict[str : List]]
        List of results for different model training runs.
        - The keys of the dictionary must be:
            * "train_loss"
            * "train_acc"
            * "test_loss"
            * "test_acc"
        - For any one element of the list, the jth element of the value
          of each key (i.e. of the list corresponding to that key)
          corresponds to the results for the (j+1)th epoch.
    model_names_lst : List[str] or None
        Contains list of names of the models whose results are in
        <results_lst>.
        - The ith element of <results_lst> is the dictionary of results
        for the model with a name given by the ith element of
        <model_names_lst>.

    Returns
    -------
    None.

    """
    max_len = 0
    # find maximum number of epochs
    for model_results in results_lst:
        if len(model_results) > max_len:
            max_len = len(model_results)
    epochs = range(1, max_len+1) # array for x-axis
    
    # Create plot
    keys = results_lst[0].keys()
    plt.figure(figsize = (15, 10))
    
    for k, key in enumerate(keys):
        plt.subplot(2,2,k+1);
    
        for i, model_results in enumerate(results_lst):
            if model_names_lst:
                model_name = model_names_lst[i]
            else:
                model_name = "Model " + str(i)
            
            plt.plot(epochs, model_results[key], label = model_name)
        
        #if "acc" in key:
        #  plt.ylim(0, 1)
        plt.xlim(epochs[0], epochs[-1])
        
        plt.title(key)
        plt.xlabel("Epochs")
        plt.legend()
    return

def compare_results_2(results_lst, model_names_lst = None):
    """
    Plots results of all the models whose results are given in <results_lst>
    on one plot so they can be seen side-by-side.
    
    There are n rows in the plot, one for each dictionary in <results_lst> 
    (n = <len(results_lst>), and 2 columns. The first column plots training
     and test loss, and the second plots the training and test accuracy.
    
    Parameters
    ----------
    results_lst : List[Dict[str : List]]
        List of results for different model training runs.
        - The keys of the dictionary must be:
            * "train_loss"
            * "train_acc"
            * "test_loss"
            * "test_acc"
        - For any one element of the list, the jth element of the value
          of each key (i.e. of the list corresponding to that key)
          corresponds to the results for the (j+1)th epoch.
    model_names_lst : List[str] or None
        Contains list of names of the models whose results are in
        <results_lst>.
        - The ith element of <results_lst> is the dictionary of results
        for the model with a name given by the ith element of
        <model_names_lst>.

    Returns
    -------
    None.

    """
    max_len = 0
    # find maximum number of epochs
    for model_results in results_lst:
        if len(model_results) > max_len:
            max_len = len(model_results)
    epochs = range(1, max_len+1) # array for x-axis
    
    # Create plot
    n = len(results_lst)
    keys_lst = [["train_loss", "test_loss", "Loss"], ["train_acc", "test_acc", "Accuracy"]]
    
    rows = n
    cols = len(keys_lst)
    
    plt.figure(figsize = (4*n, 10))
    
    plot_num = 0
    for i, model_results in enumerate(results_lst):
        for j, keys in enumerate(keys_lst):
            if model_names_lst:
                model_name = model_names_lst[i]
            else:
                model_name = "Model " + str(i)
                
            plot_num += 1
            plt.subplot(rows, cols, plot_num)
            plt.plot(epochs, model_results[keys[0]], label = keys[0])
            plt.plot(epochs, model_results[keys[1]], label = keys[1])
            plt.title(model_name + " - " + keys[2])
            plt.legend()
    return


            
            
            
    
    
    
    
