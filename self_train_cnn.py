import argparse
from logger import *
from configReader import *
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_with_cnn import MNIST_CNN_Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from custom_mnist import CustomMNIST


'''
TODO:
        add self-training wrapper
'''
def self_train(cfg, logger):
    '''contains code to self-train'''

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
    train_split_percentage = cfg["self_train"]["train_split"]
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


    #starting self-training loop
    self_train_epochs = len(val_set)//cfg["self_train"]["batch_size"]
    
    logger.info(f"number of self-train-epochs: {self_train_epochs}")
    
    self_train_running_test_acc = []

    for self_train_epoch in tqdm(range(self_train_epochs), desc = "self_train "):

        #log dataset sizes
        logger.info(f"current train_set size: {len(train_set)}")
        logger.info(f"current val_set size: {len(val_set)}")
        logger.info(f"current test_set size: {len(test_set)}")
        #assign data loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = cfg["model_params"]["train_batch_size"], shuffle = True, num_workers = 4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = cfg["model_params"]["val_batch_size"], shuffle = True, num_workers = 1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = cfg["model_params"]["test_batch_size"], shuffle = True, num_workers = 1)

        #train the model on train_set
        model.train()
        
        train_loss_value_list = []
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

            # logger.debug(f"self_train_epoch: {self_train_epoch + 1}/{self_train_epochs} : train_epoch: {train_epoch + 1} : loss: {overall_train_batch_loss}")
            train_loss_value_list.append(overall_train_batch_loss)

        # save the model
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
            self_train_running_test_acc.append(accuracy_avg)
            logger.debug(f"self_train_epoch: {self_train_epoch + 1}/{self_train_epochs} : average_accuracy: {accuracy_avg}")


        #validate the model on val_set and choose cfg["self_train"]["batch_size"] images from it and
        #move it to train_set by keeping the prediction as groundtruth label in the train_set
        model.eval()
        with torch.no_grad():
            validation_loss_val_dict = {}
            validation_loss_val_list = []
            #test on each validation image

            for input, ground_truth_labels, idx in tqdm(val_loader, desc = "val_loop   "):
                
                #TODO: check what val_loader is returning at each iteration because "list object has no attribute to at 192"
                input = input.to(device)
                ground_truth_labels = ground_truth_labels.to(device)

                predicted_labels = model(input)
                max_predicted_labels = torch.argmax(predicted_labels, dim = 1, keepdim = True)

                #find the loss
                loss = loss_criterion(predicted_labels, ground_truth_labels)
                
                #save the loss value
                validation_loss_val = float(loss.cpu().item())
                # logger.debug(f"validation loss: {validation_loss_val}")
                
                #idx is the index corresponding to the input and the label
                index_pred_dict = {}
                index_pred_dict["index"] = list(np.array(idx))[0]
                index_pred_dict["predicted_value"] = max_predicted_labels.item()
                # validation_loss_val_dict[validation_loss_val] = list(np.array(idx))[0]
                # validation_loss_val_dict[str(validation_loss_val)] = index_pred_dict
                # logger.critical(validation_loss_val)
                # logger.critical(type(str(np.float128(validation_loss_val))))
                # validation_loss_val_dict[str(np.float128(validation_loss_val))] = index_pred_dict
                temp_tuple = (str(np.float128(validation_loss_val)), list(np.array(idx))[0], max_predicted_labels.item())
                validation_loss_val_list.append(temp_tuple)

            
            #get the indices which needs to be moved
            # chosen_indices, to_be_assigned_labels = get_sample_indices(validation_loss_val_dict, cfg, logger)
            chosen_indices, to_be_assigned_labels = get_sample_indices_new(validation_loss_val_list, cfg, logger)
            chosen_indices = list(np.array(chosen_indices))
            to_be_assigned_labels = list(np.array(to_be_assigned_labels))

            logger.debug(f"chosen indices length: {len(chosen_indices)}")
            
            #increase train_set indices
            #TODO: reset below line
            train_set_indices_clone = train_set.indices.clone()
            # train_set_indices_clone = train_set.indices.copy()
            train_set_indices_clone = train_set_indices_clone.numpy()
            train_set_indices_clone = list(train_set_indices_clone)
            train_set_indices_clone.extend(chosen_indices)
            train_set.indices = torch.tensor(train_set_indices_clone)
            # logger.debug(f"before  {train_set.dataset.targets}")
            temp_before = train_set.dataset.targets.clone()
            # logger.debug(f"predlbl:{to_be_assigned_labels}")
            # assign the prediction as ground truth
            for i,val in enumerate(chosen_indices):
                chosen_idx = val
                temp_pred_val = to_be_assigned_labels[i]
                train_set.dataset.targets[chosen_idx] = torch.tensor(temp_pred_val)
            
            temp_after = train_set.dataset.targets
            # logger.debug(f"{len(train_set.dataset.targets)} - {torch.sum(temp_before == temp_after)}")
            

            #reduce val_set indices
            val_set.indices = list(np.setdiff1d(val_set.indices, chosen_indices))
            del validation_loss_val_dict

    #ending self-training loop
    logger.debug(f"self-train testing accuracy over self-train epochs: {self_train_running_test_acc}")


def last(n): 
    return n[0] #sort it with loss value

def get_sample_indices_new(val_tuple_list, cfg, logger):

    indices = []
    pred_lbls = []

    if cfg["self_train"]["loss_val_order"] == "ascending":
        logger.info(f"choosing the loss values in ascending order")
        
        sorted_keys = sorted(val_tuple_list, key = last)
        
        logger.debug(f"val dict keys length: {len(sorted_keys)}")
        logger.debug(f"type of val dict keys: {type(sorted_keys)}")


        [[indices.append(x[0]), pred_lbls.append(x[2])] for x in sorted_keys[0 : cfg["self_train"]["batch_size"]]]
        return indices, pred_lbls

    elif cfg["self_train"]["loss_val_order"] == "descending":
        logger.info(f"choosing the loss values in descending order")
        sorted_keys = sorted(val_tuple_list, key = last, reverse=True)
        
        logger.debug(f"val dict keys length: {len(sorted_keys)}")
        logger.debug(f"type of val dict keys: {type(sorted_keys)}")


        [[indices.append(x[1]), pred_lbls.append(x[2])] for x in sorted_keys[0 : cfg["self_train"]["batch_size"]]]
        return indices, pred_lbls

    elif cfg["self_train"]["loss_val_order"] == "random":
        logger.info(f"choosing the loss values in random order")
        np.random.shuffle(val_tuple_list)
        
        logger.debug(f"val dict keys length: {len(val_tuple_list)}")
        logger.debug(f"type of val dict keys: {type(val_tuple_list)}")


        [[indices.append(x[1]), pred_lbls.append(x[2])] for x in val_tuple_list[0 : cfg["self_train"]["batch_size"]]]
        return indices, pred_lbls


def get_sample_indices(argdict, cfg, logger):
    if cfg["self_train"]["loss_val_order"] == "ascending":
        logger.info(f"choosing the loss values in ascending order")
        #get the keys, here the keys are the loss values
        keys_to_sort = argdict.keys()
        keys_to_sort = list(keys_to_sort)
        logger.debug(f"val dict keys length: {len(keys_to_sort)}")
        logger.debug(f"type of val dict keys: {type(keys_to_sort)}")
        #sort the keys in ascending order
        sorted_keys = sorted(keys_to_sort)

        #choose batch_size of indices
        indices = [argdict[key]["index"] for key in sorted_keys[0 : cfg["self_train"]["batch_size"]]]
        pred_lbls = [argdict[key]["predicted_value"] for key in sorted_keys[0 : cfg["self_train"]["batch_size"]]]
        # logger.debug(f"{indices}")
        return indices, pred_lbls

    elif cfg["self_train"]["loss_val_order"] == "descending":
        logger.info(f"choosing the loss values in descending order")
        #get the keys, here the keys are the loss values
        keys_to_sort = argdict.keys()
        keys_to_sort = list(keys_to_sort)
        logger.debug(f"val dict keys length: {len(keys_to_sort)}")
        logger.debug(f"type of val dict keys: {type(keys_to_sort)}")
        #sort the keys in descending order
        sorted_keys = sorted(keys_to_sort, reverse = True)

        #choose batch_size of indices
        indices = [argdict[key]["index"] for key in sorted_keys[0 : cfg["self_train"]["batch_size"]]]
        pred_lbls = [argdict[key]["predicted_value"] for key in sorted_keys[0 : cfg["self_train"]["batch_size"]]]
        return indices, pred_lbls

    elif cfg["self_train"]["loss_val_order"] == "random":
        logger.info(f"choosing the loss values in random order")
        #get the keys, here the keys are the loss values
        permute_keys = argdict.keys()
        permute_keys = list(permute_keys)
        logger.debug(f"val dict keys length: {len(permute_keys)}")
        logger.debug(f"type of val dict keys: {type(permute_keys)}")
        
        
        #randomly permute the keys
        np.random.shuffle(permute_keys)

        #choose batch_size of indices
        indices = [argdict[key]["index"] for key in permute_keys[0 : cfg["self_train"]["batch_size"]]]
        pred_lbls = [argdict[key]["predicted_value"] for key in permute_keys[0 : cfg["self_train"]["batch_size"]]]
        return indices, pred_lbls


    # #plot necessary graphs
    # fig, ax = plt.subplots()
    # x_axis = list(torch.arange(1, len(train_loss_value_list) + 1).numpy())
    # line1, = ax.plot(x_axis, train_loss_value_list , 
    #             color='r',
    #             label='loss')

    # plt.xticks(x_axis)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # ax.legend()
    # plt.plot()
    # plt.savefig("epoch_vs_loss.png")
    # logger.info("saved the epoch_vs_loss image")


if __name__ == "__main__":
    
    #parse the arguments using argparse module
    parser = argparse.ArgumentParser(description = "self-training on MNIST self_train.py")
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
    
    
    self_train(cfg,logger)

