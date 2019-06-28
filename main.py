import argparse
from logger import *
from configReader import *
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import MNIST_Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
TODO:   
        add validation code
        check for overfitting
        test on testing data
        add self-training wrapper
'''
def runner(cfg, logger):
    '''contains code to train, validate and test on MNIST dataset'''

    logger.info("config content below")
    logger.info(cfg)

    #specify transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.5),(0.5))
        
        ]
    )

    logger.info("loading MNIST dataset")
    #get the dataset
    train_set = datasets.MNIST('./data', train = True, download = True, transform = transform)
    test_set = datasets.MNIST('./data', train = False, download = True, transform = transform)
    logger.info("loaded MNIST dataset")

    #prepare the data loader
    logger.info("preparing data loader")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg["model_params"]["batch_size"], shuffle = True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg["model_params"]["batch_size"], shuffle = True, num_workers = 4)
    logger.info("prepared data loader")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processing_on = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("processing on: " + processing_on)
    
    #declare the model
    model = MNIST_Model(num_of_input_channels = cfg["model_params"]["input_features"], num_of_output_channels = cfg["model_params"]["output_features"])

    #move the model to processing unit
    model.to(device)

    #choose an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = cfg["hyperparameters"]["learning_rate"])
    
    #define the loss function
    loss_criterion = torch.nn.CrossEntropyLoss()
    
    #set the model in train mode
    model.train()

    loss_val_list = []
    for epoch in tqdm(range(cfg["hyperparameters"]["epochs"])):
        #for each epoch
        overall_loss = 0
        for input, ground_truth_labels in tqdm(train_loader):
            #for each batch 
            input = input.to(device)

            # ground_truth_labels = ground_truth_labels.float()
            ground_truth_labels = ground_truth_labels.to(device)
            # ground_truth_labels = ground_truth_labels + 1

            input = torch.reshape(input, (cfg["model_params"]["batch_size"],-1))
            # print(input.size())
            
            #clear the gradient so they do not accumulate
            optimizer.zero_grad()

            predicted_labels = model(input)

            loss = loss_criterion(predicted_labels, ground_truth_labels)
            
            # logger.debug(loss.item())
            overall_loss += float(loss.cpu().item())
            #back-propagation
            loss.backward()

            #optimization step (descent step)
            optimizer.step()
            
            # break
        # break
        logger.debug(overall_loss)
        loss_val_list.append(overall_loss)

    #testing mode
    
    # move the model to evaluation mode to avoid backprop
    model.eval()
    with torch.no_grad():
        accuracy_total = 0
        for input, ground_truth_labels in tqdm(test_loader):
            #for each batch 
            input = input.to(device)
            ground_truth_labels = ground_truth_labels.to(device)

            input = torch.reshape(input, (cfg["model_params"]["batch_size"],-1))
            # print(input.size())
            
            predicted_labels = model(input)

            max_predicted_labels = torch.argmax(predicted_labels, dim = 1, keepdim = True)
            # print("max:", max_predicted_labels)
            # print("gt:", ground_truth_labels)

            accuracy = accuracy_score(ground_truth_labels.cpu(), max_predicted_labels.cpu())
            accuracy_total += accuracy
            
            # break
        #TODO: fix accuracy calculation, it's showing 169.299 as average accuracy
        accuracy_avg = accuracy_total / (len(test_loader)/cfg["model_params"]["batch_size"])
        logger.info("average_accuracy: " + str(accuracy_avg))




    #save the model if needed
    if cfg["project_params"]["to_save_model"]:
        torch.save(model.state_dict, cfg["project_params"]["model_save_location"])


    #plot necessary graphs
    fig, ax = plt.subplots()
    x_axis = list(torch.arange(1, len(loss_val_list) + 1).numpy())
    line1, = ax.plot(x_axis, loss_val_list , 
                color='r',
                label='loss')

    plt.xticks(x_axis)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    ax.legend()
    plt.plot()
    plt.savefig("epoch_vs_loss.png")



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
    runner(cfg,logger)
