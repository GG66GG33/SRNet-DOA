import sys
import torch
import os
import shutil
from CreateDataSet import *
from train import *
from Loss import *


warnings.simplefilter("ignore")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Main_path = r'/media/data/wtg/Pycharm_code/ULA_DOA/proposed'
    Main_Data_path = Main_path + r"/DataSet"
    Data_Scenario_path = r"/SNR-10to10"
    Loss_Data_path = Main_path + r"/Loss_data"
    Model_Param_path = Main_path + r"/Model_Param"

    Set_Overall_Seed()

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

    #######################################
    ##             Commands              ##
    #######################################
    CREATE_DATA = True
    LOAD_Train_DATA = False
    TRAIN_MODE = False


    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)

    #######################################
    ##           Data Parameters         ##
    #######################################
    N = 16    # 阵元数
    D = 2    # 信源数
    Snap = 200   # 快拍数
    Array_form = "ULA"
    SNR_Train = np.arange(-10, 12, 2)
    NumberOfSampels_Train = 20000

    mode = "exp-coherent"
    DataSet = "DataSet_RA"


    #######################################
    ##         Create Data Sets          ##
    #######################################
    if CREATE_DATA:
        Set_Overall_Seed()
        CREATE_Train_DATA = True
        print("Creating Data...")
        if CREATE_Train_DATA:
            DataSet_RA = CreateDataSet_Train(
                mode=mode,
                N=N, D=D, Snap=Snap,
                NumberOfSampels = NumberOfSampels_Train,
                SNR_values = SNR_Train,
                Real = True,
                DataSet = DataSet,
                Array_form = Array_form,
                Normalized=True,
                Save=True,
                DataSet_path = Main_Data_path + Data_Scenario_path + r"/TrainData",
                True_DOA = None
            )

    #######################################
    ##         Load Data Sets          ##
    #######################################
    if LOAD_Train_DATA:
        loss_details_line = '_{}_{}_D={}_N={}_Snap={}_SNR={}to{}'.format(
            mode, NumberOfSampels_Train, D, N, Snap, SNR_Train[0], SNR_Train[-1])
        Model_details_line = '_{}_{}_D={}_N={}_Snap={}_SNR={}to{}'.format(
            mode, NumberOfSampels_Train, D, N, Snap, SNR_Train[0], SNR_Train[-1])
        train_details_line = '_{}_{}_D={}_N={}_Snap={}_SNR={}to{}.h5'.format(
            mode, NumberOfSampels_Train, D, N, Snap, SNR_Train[0], SNR_Train[-1])

        DataSet_RA_train = Read_Data(Main_Data_path + Data_Scenario_path + r"/TrainData/DataSet_RA" + train_details_line)

        Loss_Save_Path = Loss_Data_path + r"/DataSet_RA_v5_zhengjiao" + loss_details_line
        if os.path.exists(Loss_Save_Path):
            shutil.rmtree(Loss_Save_Path)
        os.makedirs(Loss_Save_Path)

        Model_Param_Save_path = Model_Param_path + r"/DataSet_RA_v5_zhengjiao" + Model_details_line
        if os.path.exists(Model_Param_Save_path):
            shutil.rmtree(Model_Param_Save_path)
        os.makedirs(Model_Param_Save_path)

    #######################################
    ##           Training stage          ##
    #######################################
    if TRAIN_MODE:
        # Train parameters
        lr = 0.001
        Batch_size = 32
        epochs = 400
        val_size = 0.1
        Train_Simulation(Model_Train_DataSet = DataSet_RA_train,
                         epochs = epochs,
                         Batch_size = Batch_size,
                         D=D,
                         optimizer_name = "Adam",
                         lr = lr,
                         weight_decay_val = 1e-9,
                         Schedular = "ReduceLROnPlateau",
                         Loss_Save_Path = Loss_Save_Path,
                         Model_Param_Save_path = Model_Param_Save_path,
                         val_size = val_size,
                         test = True
        )
