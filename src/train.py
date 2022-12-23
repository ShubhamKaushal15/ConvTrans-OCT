from utils.data import OCTVideos
import os
from tqdm import tqdm
from cct import CCT
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train(model, num_epochs, loss_function, optimizer, model_alias, 
          train_loader, device,
          model_save_dir = os.path.join("..", "models")):
    """
    Trains the model using given hyperparameters. Saves the best model based
    on validation loss.
    Saves the list of validation losses of the model during training.
    """
    model_save_path = os.path.join(model_save_dir, model_alias)
    
    if not os.path.exists(model_save_path):
      print(f"{model_save_path} path does not exist. Creating...")
      os.makedirs(f"{model_save_path}")

    else:
        print("Model Path already exists")

        # model.to(device)
        # model.double()

    loss_file_path = os.path.join(model_save_path, "losses.info")

    if not os.path.exists(loss_file_path):
        with open(loss_file_path, 'w') as a:
            pass

    model.to(device) # get model to current device
    # model.double()

    for i in range(num_epochs):  # loop over the dataset multiple times
        
        print(f"Training epoch: {i}")

        model.train() # set model to train mode

        for _, data in enumerate(tqdm(train_loader), 0):

            # get the inputs; data is a tuple
            vid_seq, targets = data
            vid_seq = vid_seq.to(device)
            targets = targets.to(torch.float32).unsqueeze(1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # getting predictions
            outputs = model(vid_seq)

            # calculate CE loss
            loss = loss_function(outputs, targets)

            # backprop
            loss.backward()

            # update gradients
            optimizer.step()

            with open(loss_file_path, 'a') as f:
                f.write(f"{loss.item()}\n")

    return 1

def main():

    model = CCT()

    dataloader = DataLoader(OCTVideos(), batch_size=2, shuffle=True)

    learning_rate = 0.005
    cross_entropy = nn.BCEWithLogitsLoss()
    sgd_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train(model, 25, cross_entropy, sgd_optimizer, 'O-CCT_mark0', dataloader, device)

if __name__ == '__main__':
    
    main()