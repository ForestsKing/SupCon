import os
from time import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.dataset import getClassifyDataset
from exp.pretrain import ExpPretrain
from model.classifier import Classifier


class ExpClassify:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting

        exp_pretrain = ExpPretrain(self.args, self.setting)
        if self.args.loss != 'CrossEntropy':
            exp_pretrain.train()
        encoder = exp_pretrain.get_encoder()

        print('\n>>>>>>>>  initing (classify) : {}  <<<<<<<<\n'.format(setting))

        self._acquire_device()
        self._make_dirs()
        self._get_loader()
        self._get_model(encoder)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            self.device = torch.device('cuda:{}'.format(self.args.devices))
            print('Use GPU: cuda:{}'.format(self.args.devices))
        else:
            self.device = torch.device('cpu')
            print('Use CPU')

    def _make_dirs(self):
        self.data_path = self.args.data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.model_path = self.args.save_path + '/' + self.setting + '/classify/model/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.log_path = self.args.save_path + '/' + self.setting + '/classify/log/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _get_loader(self):
        train_set = getClassifyDataset(self.data_path, train=False, download=False)
        test_set = getClassifyDataset(self.data_path, train=True, download=False)

        self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=16)
        self.test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=16)

    def _get_model(self, encoder):
        self.model = Classifier(self.args, encoder).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.loss_fn = CrossEntropyLoss()

    def train(self):
        print('\n>>>>>>>>  training (classify) : {}  <<<<<<<<\n'.format(self.setting))

        for e in range(self.args.epoch):
            start = time()

            self.model.train()
            train_loss, train_label, train_pred = [], [], []
            for (batch_x, batch_y) in self.train_loader:
                self.optimizer.zero_grad()

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                train_label.append(batch_y.detach().cpu().numpy())
                train_pred.append(output.detach().cpu().numpy())

            train_loss = np.mean(train_loss)
            train_label = np.concatenate(train_label, axis=0)
            train_pred = np.concatenate(train_pred, axis=0)
            train_pred = np.argmax(train_pred, axis=1)
            train_acc = np.sum(train_pred == train_label) / len(train_label)

            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} Train ACC: {2:.4f} || Cost: {3:.6f}".format(
                e, train_loss, train_acc, end - start)
            )

            torch.save(self.model.state_dict(), self.model_path + '/' + 'checkpoint.pth')

    def test(self):
        print('\n>>>>>>>>  testing (classify) : {}  <<<<<<<<\n'.format(self.setting))
        self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth'))

        with torch.no_grad():
            self.model.eval()

            test_loss, test_label, test_pred = [], [], []
            for (batch_x, batch_y) in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)

                test_loss.append(loss.item())
                test_label.append(batch_y.detach().cpu().numpy())
                test_pred.append(output.detach().cpu().numpy())

        test_loss = np.mean(test_loss)
        test_label = np.concatenate(test_label, axis=0)
        test_pred = np.concatenate(test_pred, axis=0)
        test_pred = np.argmax(test_pred, axis=1)
        test_acc = np.sum(test_pred == test_label) / len(test_label)

        print("Test Loss: {0:.6f} || Test Acc: {1:.4f}".format(test_loss, test_acc))

        return test_acc
