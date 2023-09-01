import os
from time import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.dataset import getEncodeDataset
from model.encoder import Encoder
from model.loss import NTXentLoss, SupConLoss


class ExpPretrain:
    def __init__(self, args, setting):
        print('\n>>>>>>>>  initing (pretrain) : {}  <<<<<<<<\n'.format(setting))

        self.args = args
        self.setting = setting

        self._acquire_device()
        self._make_dirs()
        self._get_loader()
        self._get_model()

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

        self.model_path = self.args.save_path + '/' + self.setting + '/pretrain/model/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.log_path = self.args.save_path + '/' + self.setting + '/pretrain/log/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _get_loader(self):
        train_set = getEncodeDataset(self.data_path, train=False, download=self.args.download)
        test_set = getEncodeDataset(self.data_path, train=True, download=False)

        self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=16)
        self.test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=16)

    def _get_model(self):
        self.model = Encoder().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)

        if self.args.loss == 'SupCon':
            self.loss_fn = SupConLoss()
        else:
            self.loss_fn = NTXentLoss()

        torch.save(self.model.state_dict(), self.model_path + '/' + 'checkpoint.pth')

    def train(self):
        print('\n>>>>>>>>  training (pretrain) : {}  <<<<<<<<\n'.format(self.setting))

        for e in range(self.args.epoch):
            start = time()
            train_loss = []

            self.model.train()
            for (batch_x, batch_y) in self.train_loader:
                self.optimizer.zero_grad()

                batch_x = torch.cat([batch_x[0], batch_x[1]], dim=0)
                batch_x = batch_x.to(self.device)  # [batch_size * view_num, 1, 28, 28]
                batch_y = batch_y.to(self.device)  # [batch_size]

                feature = self.model(batch_x)  # [batch_size * view_num, embed_dim]

                batch_size = batch_y.shape[0]
                f1, f2 = torch.split(feature, [batch_size, batch_size], dim=0)
                feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [batch_size, view_num, embed_dim]

                if self.args.loss == 'SupCon':
                    loss = self.loss_fn(feature, batch_y)
                else:
                    loss = self.loss_fn(feature)

                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            train_loss = np.mean(train_loss)
            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} || Cost: {2:.6f}".format(e, train_loss, end - start))

            torch.save(self.model.state_dict(), self.model_path + '/' + 'checkpoint.pth')

    def get_encoder(self):
        self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth'))
        return self.model.encoder
