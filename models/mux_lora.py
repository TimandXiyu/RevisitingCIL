import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet, SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from timm.scheduler import create_scheduler
from collections import OrderedDict
from datetime import datetime
import os

# fully finetune the model at first session, and then conduct simplecil.
num_workers = 16


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.data_manager = data_manager

        test_dataset = data_manager.get_dataset(np.arange(0, 299), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.test_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):

        for i, convnet in enumerate(self._network.convnet):
            self._network.convnet[i] = convnet.to(self._device)

        self._init_train(train_loader, test_loader)

    def _init_train(self, train_loader, test_loader):
        correct, total = 0, 0
        for i, (_, inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Testing'):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs)

            # calculate entropy for each logits
            logits = torch.stack(logits, dim=1)
            entropy = torch.sum(-F.softmax(logits, dim=2) * F.log_softmax(logits, dim=2), dim=2)
            min_entropy, min_index = torch.min(entropy, dim=1)

            class_order = self.data_manager._class_order
            for i, classifier_id in enumerate(min_index):
                classifier_mapping = class_order[classifier_id*30:(classifier_id+1)*30]
                intra_pred = torch.argmax(logits[i][classifier_id])
                pred = classifier_mapping[intra_pred]



            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

        info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
            self._cur_task,
            self.args['tuned_epoch'],
            train_acc,
        )

        logging.info(info)






