import argparse
from logger import *
from configReader import *
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_with_cnn import MNIST_CNN_Model
from model import MNIST_Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from custom_mnist import CustomMNIST
from sklearn.metrics import accuracy_score

def base_line_runner(cfg, logger):
    '''contains code to train, validate and test on MNIST dataset'''

    logger.info("config content below")
    logger.info(cfg)

    #specify transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        
        ]
    )

    logger.info("loading MNIST dataset")
    #get the dataset
    initial_train_set = CustomMNIST(root = './data', train = True, download = True, transform = transform)
    test_set = CustomMNIST(root = './data', train = False, download = True, transform = transform)
    logger.info("loaded MNIST dataset")

    logger.info(f"splitting the dataset")
    #split the train-set into train and validation with some ratio
    train_split_percentage = cfg["base_line"]["train_split"]
    initial_train_set_size = len(initial_train_set)
    needed_train_set_size = int(train_split_percentage * initial_train_set_size)
    needed_val_set_size = initial_train_set_size - needed_train_set_size

    train_set, val_set = torch.utils.data.random_split(initial_train_set, (needed_train_set_size, needed_val_set_size))
    logger.info(f"dataset split")

    logger.info(f"initial train set size: {len(train_set)}")
    logger.info(f"initial val set size: {len(val_set)}")
    logger.info(f"initial test set size: {len(test_set)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processing_on = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("processing on: " + processing_on)
    
    #declare the model
    model = MNIST_CNN_Model(num_of_input_channels = cfg["model_params"]["num_of_input_channels"], num_of_output_labels = cfg["model_params"]["num_of_output_labels"])

    #move the model to processing unit
    model.to(device)

    #choose an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = cfg["hyperparameters"]["learning_rate"])
    
    #define the loss function
    loss_criterion = torch.nn.NLLLoss()

    #base-line epochs
    base_line_epochs = len(val_set)//cfg["base_line"]["batch_size"]

    logger.info(f"number of self-train-epochs: {base_line_epochs}")

    baseline_test_accs = [] 

    for base_line_epoch in tqdm(range(base_line_epochs), desc = "base_line  "):

        #assign data loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg["model_params"]["train_batch_size"], shuffle = True, num_workers = 4)
        # val_loader = torch.utils.data.DataLoader(val_set, batch_size = cfg["model_params"]["val_batch_size"], shuffle = True, num_workers = 1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg["model_params"]["test_batch_size"], shuffle = True, num_workers = 1)


        #set the model in train mode
        model.train()
        # train_loss_value_list = []
        for train_epoch in tqdm(range(cfg["hyperparameters"]["epochs"]), desc = "train_epoch"):
            #for each epoch
            overall_train_batch_loss = 0

            # logger.info(f"current train_set size: {len(train_set)}")
            for input, ground_truth_labels, train_idx in tqdm(train_loader, desc = "train_batch"):
            # for input, ground_truth_labels in tqdm(train_loader, desc = "train_batch"):
                #for each batch 
                input = input.to(device)

                # ground_truth_labels = ground_truth_labels.float()
                ground_truth_labels = ground_truth_labels.to(device)
                
                #clear the gradient so they do not accumulate
                optimizer.zero_grad()

                predicted_labels = model(input)

                loss = loss_criterion(predicted_labels, ground_truth_labels)
                
                # logger.debug(loss.item())
                overall_train_batch_loss += float(loss.cpu().item())
                #back-propagation
                loss.backward()

                #optimization step (descent step)
                optimizer.step()

            
            # train_loss_value_list.append(overall_train_batch_loss)

        #save the model if needed
        if cfg["project_params"]["to_save_model"]:
            torch.save(model.state_dict(), cfg["project_params"]["model_save_location"])

        #test the model on test_set
        model.eval()
        with torch.no_grad():
            
            accuracy_total = 0

            for input, ground_truth_labels, idx in tqdm(test_loader, desc = "test__loop "):
                #for each batch 
                input = input.to(device)
                ground_truth_labels = ground_truth_labels.to(device)
                predicted_labels = model(input)

                max_predicted_labels = torch.argmax(predicted_labels, dim = 1, keepdim = True)
                # print("max:", max_predicted_labels)
                # print("gt:", ground_truth_labels)

                accuracy = accuracy_score(ground_truth_labels.cpu(), max_predicted_labels.cpu())
                accuracy_total += accuracy
                
            #the following accuracy calculation is for batchsize = 1
            accuracy_avg = accuracy_total / (len(test_loader))
            logger.debug(f"base-line epoch :{base_line_epoch + 1}/{base_line_epochs} with current test accuracy: {accuracy_avg}")
            baseline_test_accs.append(accuracy_avg)

        #increase the train_set size
        logger.info(f"splitting the dataset")
        #split the train-set into train and validation with some ratio
        
        needed_train_set_size = len(train_set) + cfg["base_line"]["batch_size"]
        needed_val_set_size = len(val_set) - cfg["base_line"]["batch_size"]

        train_set, val_set = torch.utils.data.random_split(initial_train_set, (needed_train_set_size, needed_val_set_size))
        logger.info(f"current train set size: {len(train_set)}")
        logger.info(f"current val set size: {len(val_set)}")
        logger.info(f"current test set size: {len(test_set)}")
    
    logger.debug(f"base-line test accuracies: {baseline_test_accs}")


if __name__ == "__main__":
    
    #parse the arguments using argparse module
    parser = argparse.ArgumentParser(description="self-training on MNIST main.py")
    parser.add_argument(
        '--config',
        type = str,
        help='location of the yaml configuration file'
    )
    args = parser.parse_args()

    #get the configuration file
    config = ConfigReader(args.config)
    cfg = config.getConfig()

    #setup the logging methods
    log_config = cfg["logging"]
    log_root_obj = CustomLogger(log_config)
    logger = log_root_obj.getLogger()
    base_line_runner(cfg,logger)
