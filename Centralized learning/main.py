import argparse
import logging
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
from torch.nn.parallel import DistributedDataParallel

from centralized_trainer import CentralizedTrainer
from data_loader import load_partition_data_fed_wisdm2011



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='simple_mlp', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--data_parallel', type=int, default=0,
                        help='if distributed training')

    parser.add_argument('--dataset', type=str, default='fed_wisdm2011', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=1, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=300, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=100, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=0,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_train_acc_report', type=int, default=10,
                        help='the frequency of training accuracy report')

    parser.add_argument('--frequency_of_test_acc_report', type=int, default=1,
                        help='the frequency of test accuracy report')

    parser.add_argument('--gpu_server_num', type=int, default=0,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=0,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--gpu_util', type=str, default='0',
                        help='gpu utils')

    parser.add_argument('--local_rank', type=int, default=0,
                        help='given by torch.distributed.launch')


    args = parser.parse_args()
    return args


def load_data(dataset_name): #modified
    args_batch_size = args.batch_size
    if dataset_name == "fed_wisdm2011":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_fed_wisdm2011(batch_size=args.batch_size,fold_idx=2)

        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]


    return dataset

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_nodes=90, hidden_nodes=40, output_nodes=6):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)

        # Initialize weights similar to your InitialWeightMax
        nn.init.uniform_(self.fc1.weight, -0.9, 0.9)
        nn.init.uniform_(self.fc2.weight, -0.9, 0.9)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None

    if model_name == "simple_mlp":
        logging.info("Simple MLP (90-40-6)")
        model = SimpleMLP(input_nodes=90, hidden_nodes=40, output_nodes=output_dim)

    return model


if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.world_size = len(args.gpu_util.split(','))
    worker_number = 1
    process_id = 0

    if args.data_parallel == 1:
        # torch.distributed.init_process_group(
        #         backend="nccl", world_size=args.world_size, init_method="env://")
        torch.distributed.init_process_group(
                backend="nccl", init_method="env://")
        args.rank = torch.distributed.get_rank()
        gpu_util = args.gpu_util.split(',')
        gpu_util = [int(item.strip()) for item in gpu_util]
        # device = torch.device("cuda", local_rank)
        torch.cuda.set_device(gpu_util[args.rank])
        process_id = args.rank
    else:
        args.rank = 0

    logging.info(args)

    # customize the process name
    str_process_name = "Fedml (single centralized):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        # logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    logging.info("process_id = %d, size = %d" % (process_id, args.world_size))

    # load data
    dataset = load_data("fed_wisdm2011")
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
    print("class_num= ",class_num)
    model = create_model(args, model_name=args.model, output_dim=dataset[-1])

    device = torch.device("cpu")
    single_trainer = CentralizedTrainer(dataset, model, device, args)
    single_trainer.train()
