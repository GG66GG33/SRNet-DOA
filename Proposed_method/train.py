import torch
import warnings
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from model import *
from Loss import *

warnings.simplefilter("ignore")
plt.close('all')

def Set_Overall_Seed(SeedNumber = 42):
  random.seed(SeedNumber)
  np.random.seed(SeedNumber)
  torch.manual_seed(SeedNumber)

Set_Overall_Seed()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Train_Simulation(Model_Train_DataSet, epochs,
                   Batch_size, D, optimizer_name, lr, weight_decay_val, Schedular,
                   Loss_Save_Path, Model_Param_Save_path, val_size, test):

    Set_Overall_Seed()

    ## Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)

    ############################
    ### Model initialization ###
    ############################
    model = Proposed_model(D)
    model = model.to(device)

    ############################
    ### Optimizer  ###
    ############################
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_val)

    if Schedular == "ReduceLROnPlateau":
        lr_decay = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=10, verbose=True)

    ############################
    ###          Loss        ###
    ############################
    criterion = orthogonality_loss()
    ############################
    ###   Data Organization  ###
    ############################

    ## Split data into Train and Validation
    Train_DataSet, Valid_DataSet = train_test_split(Model_Train_DataSet, test_size=val_size, shuffle=True)
    print("Training DataSet size", len(Train_DataSet))
    print("Validation DataSet size", len(Valid_DataSet))

    ## Transform Training Datasets into DataLoader Object
    Train_data = torch.utils.data.DataLoader(Train_DataSet,
                                             batch_size=Batch_size,
                                             shuffle=True,
                                             drop_last=False)
    Valid_data = torch.utils.data.DataLoader(Valid_DataSet,
                                             batch_size=1,
                                             shuffle=False,
                                             drop_last=False)


    ############################
    ###     Train Model      ###
    ############################
    model = train_model(epochs=epochs, model=model, criterion = criterion,
                                                        optimizer = optimizer, lr_decay = lr_decay,
                                                        Schedular_name = Schedular,
                                                        Train_data = Train_data,
                                                        Valid_data = Valid_data,
                                                        Loss_Save_Path =Loss_Save_Path,
                                                        Model_Param_Save_path=Model_Param_Save_path)

def train_model(epochs, model, criterion, optimizer, lr_decay, Schedular_name,
                Train_data, Valid_data,
                Loss_Save_Path, Model_Param_Save_path):
    writer_train = SummaryWriter(Loss_Save_Path+ "/Train")
    writer_val = SummaryWriter(Loss_Save_Path + "/Val")
    writer_lr = SummaryWriter(Loss_Save_Path + "/lr")
    Val_loss_min = 99999.0
    for epoch in range(epochs):
        Train_loss = one_epoch(model, criterion, optimizer, lr=optimizer.param_groups[0]['lr'], is_train=True,
                               epoch_now=epoch, epoch_all=epochs, dataloader=Train_data)
        Val_loss = one_epoch(model, criterion, optimizer, lr=optimizer.param_groups[0]['lr'], is_train=False,
                             epoch_now=epoch, epoch_all=epochs, dataloader=Valid_data)

        if Schedular_name == "ReduceLROnPlateau":
            lr_decay.step(Val_loss)

        writer_train.add_scalar("Loss", Train_loss, epoch)
        writer_val.add_scalar("Loss", Val_loss, epoch)
        writer_lr.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model, Model_Param_Save_path + "/model_{}.pth".format(epoch + 1))
        if Val_loss_min > Val_loss:
            Val_loss_min = Val_loss
            torch.save(model, Model_Param_Save_path + "/model_best.pth")

    writer_train.close()
    writer_val.close()
    writer_lr.close()

    return model

def one_epoch(model, criterion, optimizer, lr, is_train, epoch_now, epoch_all, dataloader):
    if is_train == True:
        model.train()
        desc = f'\033[1;34;40m[Training] Epoch {epoch_now + 1}/{epoch_all} mean loss:'
    else:
        model.eval()
        desc = f'\033[1;32;40m[Validation] Epoch {epoch_now + 1}/{epoch_all} mean loss:'
    mean_loss = torch.zeros(1).to(device)
    dataloader = tqdm(dataloader, desc=desc, file=sys.stdout)
    for step, data in enumerate(dataloader):
        x_batch, y_batch = data
        x_batch = x_batch.float()
        outputs_train = model(x_batch.to(device))
        outputs_train = (outputs_train[:, 0, :, :].type(torch.complex64) + 1j * outputs_train[:, 1, :, :].type(
            torch.complex64)).unsqueeze(1)
        y_batch = (y_batch[:, 0, :, :].type(torch.complex64) + 1j * y_batch[:, 1, :, :].type(
            torch.complex64)).unsqueeze(1)
        loss = criterion(outputs_train, y_batch.to(device))
        mean_loss = (mean_loss * step + loss.item()) / (step + 1)
        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dataloader.desc = "[Training epoch {}] mean loss:{}".format(epoch_now, round(mean_loss.item(), 6)) + ' lr:{}'.format(lr)
        else:
            dataloader.desc = "[Val epoch {}] mean loss:{}".format(epoch_now, round(mean_loss.item(), 6)) + ' lr:{}'.format(lr)
    return mean_loss.item()


