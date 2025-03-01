import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
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
        if 'resnet' in args['convnet_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self. batch_size = 128
            self.init_lr = args["init_lr"] if args["init_lr"] is not None else  0.01
        else:
            self._network = SimpleVitNet(args, True)
            self. batch_size = args["batch_size"]
            self. init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.reset_fc(data_manager.get_task_size(self._cur_task))
        # self._network.reset_lora()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        
        self._network.to(self._device)

        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

        if self.args['optimizer']=='sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer']=='adam':
            optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self.save_lora_fc()

    def save_lora_fc(self):
        lora_fc_params = OrderedDict()
        for name, param in self._network.named_parameters():
            if ('lora' in name or 'fc' in name) and 'fc1' not in name and 'fc2' not in name:
                lora_fc_params[name] = param
        # create sub folders named after the date and task id
        cur_date = self.args['current_date']
        task_id = self._cur_task
        # create the folder if not exist
        if not os.path.exists(self.args['save_path'] + '/{}'.format(cur_date)):
            os.makedirs(self.args['save_path'] + '/{}'.format(cur_date))
        torch.save(lora_fc_params, self.args['save_path'] + '/{}/lora_fc_params_task_{}.pth'.format(cur_date, task_id))

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        for _, epoch in enumerate(range(self.args['tuned_epoch'])):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                # offset targets between 0-29
                targets = targets - self._known_classes

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # if epoch % 5 == 0:
            #     info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
            #         self._cur_task,
            #         epoch + 1,
            #         self.args['tuned_epoch'],
            #         losses / len(train_loader),
            #         train_acc,
            #     )
            # else:
            test_acc = self._compute_accuracy(self._network, test_loader, offset_label=self._known_classes)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )

            logging.info(info)

    




