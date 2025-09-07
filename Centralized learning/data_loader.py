''' using the raw WISDM_ar_v1.1_raw.txt dataset '''

import json
import logging
import torch
import torch.utils.data as data
from tqdm import tqdm
import random

random.seed(0)
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_BATCH_SIZE = 300

_USERS = 'users'
_USER_DATA = "user_data"

def load_partition_data_fed_wisdm2011(data_dir=None, batch_size=1,fold_idx=1):
    print("load_partition_data_fed_wisdm2011 START")
    print("batch_size", batch_size)
    print("fold_idx", fold_idx)
    train_file_path = 'data/train/' + f"fold_{fold_idx}_train.json"
    test_file_path = 'data/test/' + f"fold_{fold_idx}_test.json"
    with open(train_file_path, 'r') as train_f, open(test_file_path, 'r') as test_f:
        train_data = json.load(train_f)
        test_data = json.load(test_f)

        client_ids_train = train_data[_USERS]
        client_ids_test = test_data[_USERS]
        client_num = len(client_ids_train)

        full_x_train = torch.tensor([], dtype=torch.float32)
        full_y_train = torch.tensor([], dtype=torch.int64)
        full_x_test = torch.tensor([], dtype=torch.float32)
        full_y_test = torch.tensor([], dtype=torch.int64)
        train_data_local_dict = {}
        test_data_local_dict = {}

        # Process train data
        with tqdm(total=len(client_ids_train), desc='train data') as pbar:
            for i, client_id in enumerate(client_ids_train):
                client_x = train_data[_USER_DATA][str(client_id)]['x']
                client_y = train_data[_USER_DATA][str(client_id)]['y']

                # client_x_win, client_y_win = reshape_to_windows(client_x, client_y)
                client_x_win = torch.tensor(client_x, dtype=torch.float32)
                client_y_win = torch.tensor(client_y, dtype=torch.int64)
                train_ds = data.TensorDataset(client_x_win, client_y_win)
                train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
                train_data_local_dict[i] = train_dl

                full_x_train = torch.cat((full_x_train, client_x_win), 0)
                full_y_train = torch.cat((full_y_train, client_y_win), 0)
                pbar.update(1)

        # Process test data
        with tqdm(total=len(client_ids_test), desc='test data') as pbar1:
            for i, client_id in enumerate(client_ids_test):
                client_x = test_data[_USER_DATA][str(client_id)]['x']
                client_y = test_data[_USER_DATA][str(client_id)]['y']

                # Convert to tensors
                client_x_win = torch.tensor(client_x, dtype=torch.float32)
                client_y_win = torch.tensor(client_y, dtype=torch.int64)

                test_ds = data.TensorDataset(client_x_win, client_y_win)
                test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
                test_data_local_dict[i] = test_dl

                full_x_test = torch.cat((full_x_test, client_x_win), 0)
                full_y_test = torch.cat((full_y_test, client_y_win), 0)
                pbar1.update(1)

        # Global datasets
        train_ds = data.TensorDataset(full_x_train, full_y_train)
        test_ds = data.TensorDataset(full_x_test, full_y_test)
        train_data_global = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_data_global = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        train_data_num = len(train_data_global.dataset)
        test_data_num = len(test_data_global.dataset)
        print("train_data_num:", train_data_num)
        print("test_data_num:", test_data_num)
        data_local_num_dict = {i: len(train_data_local_dict[i].dataset) for i in train_data_local_dict}
        output_dim = 6  # WISDM classes

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim



