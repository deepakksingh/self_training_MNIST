import argparse
from logger import *
from configReader import *
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import MNIST_Model
from tqdm import tqdm
from sklearn.metrics import accuracy_score

'''
TODO:
        add inference code to check the predictions
'''

def inference(cfg, logger):
    '''contains code to inference the MNIST dataset from the saved model'''

    logger.info("inference module started")

    #check whether the model was saved
    if cfg["project_params"]["to_save_model"]:
        #if model was saved then testing can be done using the saved model
        logger.info("using the saved model for testing")

        #check for available processing unit
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processing_on = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("processing on: " + str(device))

        #load the saved model and move it to the processing unit
        model = MNIST_Model(num_of_input_channels = cfg["model_params"]["input_features"], num_of_output_channels = cfg["model_params"]["output_features"])
        model.load_state_dict(torch.load(cfg["project_params"]["model_save_location"], map_location = processing_on))
        model.to(device)

        #specify transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #transforms.Normalize((0.5),(0.5))
            
            ]
        )

        #prepare the dataset and the dataloader
        test_set = datasets.MNIST('./data', train = False, download = True, transform = transform)
        logger.info("loaded MNIST test dataset")

        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 4)
        logger.info("prepared test data loader")

        #move the model to eval mode
        model.eval()
        
        with torch.no_grad():
            # accuracy_total = 0
            resulting_matches = []
            
            for input, ground_truth_labels in tqdm(test_loader):
                #for each batch 
                input = input.to(device)
                ground_truth_labels = ground_truth_labels.to(device)

                input = torch.reshape(input, (1,-1))
                # print(input.size())
                
                predicted_labels = model(input)

                max_predicted_labels = torch.argmax(predicted_labels, dim = 1, keepdim = True)
                
                # accuracy = accuracy_score(ground_truth_labels.cpu(), max_predicted_labels.cpu())
                # accuracy_total += accuracy


                resulting_matches.append(
                    [
                    ground_truth_labels.cpu().item(),
                    max_predicted_labels.cpu().item(),
                    torch.squeeze(ground_truth_labels.cpu() == max_predicted_labels.cpu()).item()
                    ]
                )

                
                # break
            # accuracy_avg = accuracy_total / (len(test_loader))
            # logger.info("average_accuracy: " + str(accuracy_avg))

            resulting_matches = torch.tensor(resulting_matches)
            
            torch.set_printoptions(profile = "full")
            print(resulting_matches)
            torch.set_printoptions(profile = "default")



if __name__ == "__main__":
    #parse the arguments using argparse module
    parser = argparse.ArgumentParser(description="testing module of self-training on MNIST")
    parser.add_argument(
        '--config',
        type = str,
        help = "location of the yaml configuration file"
    )

    args = parser.parse_args()

    #get the configuration file
    config = ConfigReader(args.config)
    cfg = config.getConfig()

    #setup the logging methods
    log_config = cfg["logging"]
    log_root_obj = CustomLogger(log_config)
    logger = log_root_obj.getLogger()

    inference(cfg, logger)