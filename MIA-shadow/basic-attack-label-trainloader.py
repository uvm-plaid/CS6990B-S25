# This code is modified by ceskalka for exploring a simple membership inference attack.
# See comments in attack_data and main functions for relevant discussion.
#
# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import random
import numpy as np
import sys
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from opacus import PrivacyEngine
from tqdm import tqdm

config_file = './../../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

sys.path.append(src_dir)
sys.path.append(root_dir)
# print(root_dir)
# print(src_dir)
sys.path.append(os.path.join(src_dir, 'attack'))
sys.path.append(os.path.join(src_dir, 'models'))
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from purchase import PurchaseClassifier
# ces attack model
from ces_attack import CesMiaClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This function builds an attack dataset for training the attack model.
# Given a model, train_data, and test_data, this function will compute
# the softmax outer layer and append it to each input sample for inclusion
# in the attack dataset, with label 0 if the sample comes from test_data,
# and 1 if the sample comes from train_data.
def attack_data(model, train_data, train_labels, test_data, test_labels):
    model.eval()
    
    train_inputs = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_outputs = F.softmax(model(train_inputs),dim=1)
    
    test_inputs = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_outputs = F.softmax(model(test_inputs),dim=1)  

    zerovec = np.full(len(test_data), 0)
    onevec = np.full(len(train_data), 1)

    pre_xta1 = torch.cat((train_outputs, test_outputs)).detach().numpy()
    pre_xta2 = np.reshape(np.vstack((train_labels, test_labels)), (2 * train_labels.size))
    pre_xta2_onehot = (F.one_hot(torch.from_numpy(pre_xta2).type(torch.LongTensor), num_classes=100)).numpy()
    data_a = np.hstack((pre_xta1, pre_xta2_onehot))
    ## Next two lines include victim/shadow model inputs as attack model features.
    # pre_xta3 = np.vstack((train_data, test_data))
    # data_a = np.hstack((data_a, pre_xta3))
    label_a = np.hstack((onevec,zerovec))

    return data_a, label_a

def test(model, test_data, test_label, batch_size):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t = int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        # measure data loading time
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data_tensor[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = test_label_tensor[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # DISABLED conf calculation due to conflict with attack training
        #infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        #conf = np.mean(np.max(infer_np[batch_ind*batch_size:end_idx], axis = 1))
        #confs.update(conf, inputs.size()[0])        

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1,2))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    print('Average Testing Accuracy: {:.4f}.'.format(top1.avg))
    sys.stdout.flush()
        
def train(model, train_data, train_label, optimizer, num_epochs, learning_rate, batch_size):
    model.train()
    criterion = nn.CrossEntropyLoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_label, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        # for inputs, labels in train_loader:
        for _batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track the loss
            running_loss += loss.item()
        
        # Print the average loss for the epoch
        # if report_loss: 
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    
    print("Training complete.")

def main():
    parser = argparse.ArgumentParser(description='undefend training for Purchase dataset')
    parser.add_argument('--attack_epochs', type = int, default = 45, help = 'attack epochs in NN attack')
    parser.add_argument('--classifier_epochs', type = int, default = 30, help = 'classifier epochs')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--lr', type = float, default = .001, help = 'learning rate')
    parser.add_argument('--num_class', type = int, default = 100, help = 'num class')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs
    classifier_epochs = args.classifier_epochs
    lr = args.lr

    DATASET_PATH = os.path.join(root_dir, 'purchase',  'data')
    checkpoint_path = os.path.join(root_dir, 'purchase', 'checkpoints', 'undefend')
    # print(checkpoint_path)
    
    train_data_v = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data_v.npy'))
    train_label_v = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label_v.npy'))
    test_data_v = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data_v.npy'))
    test_label_v = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label_v.npy'))

    model_v = PurchaseClassifier()

    # optimizer = optim.SGD(model_v.parameters(), lr=lr, momentum=0)
    optimizer = optim.Adam(model_v.parameters(), lr=lr)
    
    print("VICTIM TRAINING")
    train(model_v, train_data_v, train_label_v, optimizer, classifier_epochs, lr, batch_size)

    train_data_s = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data_s.npy'))
    train_label_s = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label_s.npy'))
    test_data_s = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data_s.npy'))
    test_label_s = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label_s.npy'))

    model_s = PurchaseClassifier()

    optimizer = optim.Adam(model_s.parameters(), lr=lr)

    print("SHADOW TRAINING")
    train(model_s, train_data_s, train_label_s, optimizer, classifier_epochs, lr, batch_size)
    
    train_data_a, train_label_a = attack_data(model_s, train_data_s, train_label_s, test_data_s, test_label_s)
    test_data_a, test_label_a = attack_data(model_v, train_data_v, train_label_v, test_data_v, test_label_v)
    
    model_a = CesMiaClassifier(200,512,1028,128,2)

    optimizer = optim.Adam(model_a.parameters(), lr=lr)

    print("ATTACK TRAINING")
    train(model_a, train_data_a, train_label_a, optimizer, attack_epochs, lr, batch_size)

    print("==================")
    print("TESTING ACCURACIES")
    print("==================")
    print("VICTIM")
    test(model_v, test_data_v, test_label_v, batch_size)
    print("SHADOW")
    test(model_s, test_data_s, test_label_s, batch_size)
    print("ATTACK ON VICTIM")
    test(model_a, test_data_a, test_label_a, batch_size)
    
if __name__ == '__main__':
    main()
