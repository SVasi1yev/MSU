import pickle
import time
from IPython.display import clear_output
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Класс простого датасета для обучения и применения модели на PyTorch.
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Класс простого загрузчика данных для обучения и применения модели на PyTorch.
class SimpleDataloader:
    def __init__(self, X, y, batch_size=2**13, shuffle=True):
        order = np.arange(X.shape[0], dtype=int)
        if shuffle:
            np.random.shuffle(order)
        self.X = X[order]
        self.y = y[order]
        self.batch_size = batch_size

    def __iter__(self):
        self.batch_num = 0
        return self
    
    def __next__(self):
        if (self.batch_num * self.batch_size) >= len(self.X):
            raise StopIteration
        batch_X = self.X[self.batch_num * self.batch_size: (self.batch_num + 1) * self.batch_size]
        batch_y = self.y[self.batch_num * self.batch_size: (self.batch_num + 1) * self.batch_size]
        self.batch_num += 1
        return batch_X, batch_y
    

# Класс загрузчика данных для обучения и применения модели на PyTorch.
# Дублирование объектов n_dup раз, каждый раз с вероятностью, 
# определяемой функцией p_func(z). Батч формируется из объектов с z < z_thr и z > z_thr 
# пропорционально распределению объектов в выборке
class HZ_dataloader_new:
    def __init__(self, X, y, p_func, n_dup, z_thr=5, batch_size=2**13, shuffle=True, device=torch.device('cpu')):
        self.device = device
        self.X = X
        self.y = y
        y_cpu = y.cpu().numpy()
        self.all_idx = np.arange(len(y), dtype=int)
        p = p_func(y_cpu)
        for i in range(n_dup):
            self.all_idx = np.concatenate(
                [self.all_idx] +
                [np.arange(len(y), dtype=int)[np.random.uniform(low=0.0, high=1.0, size=len(y)) <= p]]
            )
        
        self.z_thr = z_thr
        self.all_idx = torch.tensor(self.all_idx, device=device)
        self.low_z_idx = self.all_idx[self.y[self.all_idx] < self.z_thr]
        self.high_z_idx = self.all_idx[self.y[self.all_idx] >= self.z_thr]
        self.batch_size = batch_size
        self.low_z_frac = len(self.low_z_idx) / len(self.all_idx)
        self.low_z_batch_size = int(self.batch_size * self.low_z_frac)
        self.high_z_batch_size = self.batch_size - self.low_z_batch_size
        self.shuffle = shuffle
        
        self.low_z_idx_idx = np.arange(len(self.low_z_idx))
        self.high_z_idx_idx = np.arange(len(self.high_z_idx))
        if self.shuffle:
            np.random.shuffle(self.low_z_idx_idx)
            np.random.shuffle(self.high_z_idx_idx)
        self.low_z_idx_idx = torch.tensor(self.low_z_idx_idx, device=self.device)
        self.high_z_idx_idx = torch.tensor(self.high_z_idx_idx, device=self.device)
             
    def __iter__(self):
        self.batch_num = 0
        return self
        
    def __next__(self):
        if self.low_z_batch_size * self.batch_num >= len(self.low_z_idx_idx) or self.high_z_batch_size * self.batch_num >= len(self.high_z_idx_idx):
            raise StopIteration
        batch_X = torch.concatenate([
            self.X[self.low_z_idx[self.low_z_idx_idx[self.low_z_batch_size * self.batch_num: self.low_z_batch_size * (self.batch_num + 1)]]],
            self.X[self.high_z_idx[self.high_z_idx_idx[self.high_z_batch_size * self.batch_num: self.high_z_batch_size * (self.batch_num + 1)]]]
        ])
        batch_y = torch.concatenate([
            self.y[self.low_z_idx[self.low_z_idx_idx[self.low_z_batch_size * self.batch_num: self.low_z_batch_size * (self.batch_num + 1)]]],
            self.y[self.high_z_idx[self.high_z_idx_idx[self.high_z_batch_size * self.batch_num: self.high_z_batch_size * (self.batch_num + 1)]]]
        ])
        self.batch_num += 1
        return batch_X, batch_y
    

# Класс многослойного полносвязного перцептрона для оценки смеси нормальных распределений. 
# sizes - количество и размены слоёв. 
# p - параметр Dropout.
class MLP_GMM(nn.Module):
    def __init__(self, sizes, p):
        super().__init__()
        self.sizes = sizes
        layers = []
        for i in range(1, len(sizes)-1):
            if i > 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p))
            layers.append(nn.Linear(self.sizes[i-1], self.sizes[i], bias=False))
            layers.append(nn.BatchNorm1d(self.sizes[i]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p))
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)
        self.pi = nn.Linear(self.sizes[-2], self.sizes[-1])
        self.init_weights(self.pi)
        # self.softmax = nn.Softmax(dim=1)
        self.mu = nn.Linear(self.sizes[-2], self.sizes[-1])
        self.init_weights(self.mu)
        self.sigma = nn.Linear(self.sizes[-2], self.sizes[-1])
        self.init_weights(self.sigma)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.n_gauss = self.sizes[-1]
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                          
    def forward(self, x, train_m):
        self.train(train_m)
        x = self.layers(x)
        pi, mu, sigma = nn.functional.gumbel_softmax(self.pi(x), tau=1, dim=-1) + 1e-10, self.relu(self.mu(x)), self.elu(self.sigma(x)) + 1 + 1e-10
        return pi, mu, sigma
    
    def run_epoch(self, dataloader, optimizer=None, scheduler=None, loss_f=None):
        train_m = (optimizer is not None) and (loss_f is not None)
        true, pi, mu, sigma = [], [], [], []
        for X, y in dataloader:
            if train_m:
                optimizer.zero_grad()
            batch_pi, batch_mu, batch_sigma = self(X, train_m)
            if train_m:
                loss = loss_f(y, batch_pi, batch_mu, batch_sigma)
                loss.backward()
                optimizer.step()
            true.append(y.cpu().detach().numpy())
            pi.append(batch_pi.cpu().detach().numpy())
            mu.append(batch_mu.cpu().detach().numpy())
            sigma.append(batch_sigma.cpu().detach().numpy())
        if train_m:
            scheduler.step()
        true = np.concatenate(true)
        pi = np.vstack(pi)
        mu = np.vstack(mu)
        sigma = np.vstack(sigma)
        return true, pi, mu, sigma
    
    @classmethod
    def get_dataloader(cls, X, y, batch_size=2**13, shuffle=True):
        order = np.arange(X.shape[0], dtype=int)
        if shuffle:
            np.random.shuffle(order)
        dataset = SimpleDataset(X[order], y[order])
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        return dataloader
    

# Класс для сохранения модели глубокого ансамбля для оценки смеси нормальных распределений.
class DeepEnsemble_GMM_ObjToSave:
    def __init__(self, BaseModel, base_model_args, M, device, models):
        self.BaseModel = BaseModel
        self.base_model_args = base_model_args
        self.M = M
        self.device = device
        self.models = models


# Класс модели глубокого ансамбля для оценки смеси нормальных распределений. 
# BaseModel - класс базовой модели для оценки смеси нормальных распрееделений. 
# base_model_args - dict с парамерами для инициализации базовых моделей. 
# M - количество моделей в ансамбле. 
# device - устройство, на котором будет проихводиться обучение и предскзание.
class DeepEnsemble_GMM:
    def __init__(self, BaseModel, base_model_args, M, device=torch.device('cpu')):
        self.BaseModel = BaseModel
        self.base_model_args = base_model_args
        self.M = M
        self.device = device
        self.models = []
        for i in range(M):
            self.models.append(BaseModel(**base_model_args).to(self.device))
        
# Функция ошибки для оценки смеси нормальных распределений.
    def loss(self, y, pi, mu, sigma):
        if not isinstance(y, torch.Tensor):
            y, pi, mu, sigma = torch.Tensor(y), torch.Tensor(pi), torch.Tensor(mu), torch.Tensor(sigma)
        comp_prob = - torch.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * torch.pow((y.view(-1, 1) - mu) / sigma, 2)
        mix = torch.log(pi)
        res = torch.logsumexp(comp_prob + mix, dim=-1)
        return torch.mean(-res)

# Метод для сохранения модели в формате pickle.    
    def save_pickle(self, file):
        obj = DeepEnsemble_GMM_ObjToSave(self.BaseModel, self.base_model_args, self.M, self.device, self.models)
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

# Метод для загрузки модели.    
    @classmethod
    def load_pickle(cls, file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        ens = DeepEnsemble_GMM(obj.BaseModel, obj.base_model_args, obj.M, obj.device)
        ens.models = copy.deepcopy(obj.models)
        return ens

# Метод для обучения моедли.
# TrainDataLoader - Класс загрузчика данных для обучения.
# train_dataloader_params - параметры для инициализации загрузчика данных для обучения.
# TestDataLoader - Класс загрузчика данных для валидации.
# test_dataloader_params - параметры для инициализации загрузчика данных для валидации. 
# epochs - число эпох обучения. 
# optimizer - класс алгоритма оптимизации.
# optimizer_args - параметры для инициализации алгоритма оптимизации. 
# scheduler - класс управления алгоритмом оптимизации. 
# scheduler_args - параметры для инициализации объекта управления алгоритмом оптимизации. 
# verbose - флаг отображения информации об обучении.
# metrics - список метрик, которые должны быть подсчитаны на каждой эпохе обучения. metric_func(true, pred).
    def fit(
        self, 
        TrainDataloader, train_dataloader_params, 
        TestDataloader=None, test_dataloader_params={},
        epochs=5,
        optimizer=torch.optim.Adam, optimizer_args={'lr': 0.0005, 'weight_decay': 0.0001}, 
        scheduler=torch.optim.lr_scheduler.ExponentialLR, scheduler_args={'gamma': 0.9},
        verbose=True, metrics=[]
    ):
        # X_train_tensor = torch.tensor(X_train, device=torch.device(self.device), dtype=torch.float)
        # y_train_tensor = torch.tensor(y_train, device=torch.device(self.device), dtype=torch.float)
        # X_test_tensor = torch.tensor(X_test, device=torch.device(self.device), dtype=torch.float)
        # y_test_tensor = torch.tensor(y_test, device=torch.device(self.device), dtype=torch.float)
        
        # test_dataloader = self.BaseModel.get_dataloader(
        #     X_test_tensor, y_test_tensor,
        #     batch_size=batch_size, shuffle=shuffle
        # )
        
        test_m = TestDataloader is not None
        if test_m:
            test_dataloader = TestDataloader(**test_dataloader_params)
        
        optimizers = []
        schedulers = []
        for model in self.models:
            optimizers.append(optimizer(model.parameters(), **optimizer_args))
            schedulers.append(scheduler(optimizers[-1], **scheduler_args))

        train_metric_vals = {metric: [] for metric in metrics}
        train_losses = []
        if test_m:
            test_metric_vals = {metric: [] for metric in metrics}
            test_losses = []
        else:
            test_metric_vals = None
            test_losses = None
    
        for epoch in range(epochs):
            start = time.time()
            
            cur_lr = schedulers[0].get_last_lr()
            
            # train_dataloader = HZ_dataloader_new(
            #     X_train_tensor, y_train_tensor, p_func, n_dup, z_thr=z_thr,
            #     batch_size=batch_size, shuffle=shuffle, device=self.device
            # )

            train_dataloader = TrainDataloader(**train_dataloader_params)

            epoch_pi, epoch_mu, epoch_sigma = [], [], []
            epoch_losses = []
            for i, model in enumerate(self.models):
                epoch_true, pi, mu, sigma = model.run_epoch(train_dataloader, optimizers[i], schedulers[i], self.loss)
                epoch_pi.append(pi)
                epoch_mu.append(mu)
                epoch_sigma.append(sigma)
                epoch_losses.append(self.loss(epoch_true, pi, mu, sigma).item())
            epoch_pi = np.concatenate(epoch_pi, axis=1) / len(self.models)
            epoch_mu = np.concatenate(epoch_mu, axis=1)
            epoch_sigma = np.concatenate(epoch_sigma, axis=1)
            epoch_p = (1 / (epoch_sigma * np.sqrt(2 * np.pi))) * epoch_pi
            mode = self.mode_in_mus(epoch_pi, epoch_mu, epoch_sigma)
            mode_01 = epoch_mu[np.arange(epoch_mu.shape[0]), np.argmax(epoch_p, axis=1)]
            mu = np.sum(epoch_mu * epoch_pi, axis=1)
            sigma = np.sum((epoch_sigma ** 2) * epoch_pi, axis=1) + np.sum((epoch_mu - mu.reshape(-1, 1))**2 * epoch_pi, axis=1)
            
            train_losses.append(epoch_losses)
            for metric in metrics:
                train_metric_vals[metric].append(metrics[metric](epoch_true, mode))
                
            #TEST
            if test_m:     
                epoch_pi, epoch_mu, epoch_sigma = [], [], []
                epoch_losses = []
                for i, model in enumerate(self.models):
                    epoch_true, pi, mu, sigma = model.run_epoch(test_dataloader)
                    epoch_pi.append(pi)
                    epoch_mu.append(mu)
                    epoch_sigma.append(sigma)
                    epoch_losses.append(self.loss(epoch_true, pi, mu, sigma).item())
                epoch_pi = np.concatenate(epoch_pi, axis=1) / len(self.models)
                epoch_mu = np.concatenate(epoch_mu, axis=1)
                epoch_sigma = np.concatenate(epoch_sigma, axis=1)
                epoch_p = (1 / (epoch_sigma * np.sqrt(2 * np.pi))) * epoch_pi
                mode = self.mode_in_mus(epoch_pi, epoch_mu, epoch_sigma)
                mode_01 = epoch_mu[np.arange(epoch_mu.shape[0]), np.argmax(epoch_p, axis=1)]
                mu = np.sum(epoch_mu * epoch_pi, axis=1)
                sigma = np.sum((epoch_sigma ** 2) * epoch_pi, axis=1) + np.sum((epoch_mu - mu.reshape(-1, 1))**2 * epoch_pi, axis=1)

                test_losses.append(epoch_losses)
                for metric in metrics:
                    test_metric_vals[metric].append(metrics[metric](epoch_true, mode))    
                
            if verbose and epoch > 0:
                clear_output(True)
                print(f'Device: {self.device}')
                print(f'Test mode: {test_m}')
                print('=' * 40)
                print(f'EPOCH #{epoch+1}/{epochs}:')
                print(f'Learning rate: {round(cur_lr[0], 8)}')
                print('-' * 40)

                print(f'Trian losses: {[round(l, 5) for l in train_losses[-1]]}')
                print(f'AVG train loss: {round(np.mean(train_losses[-1]), 5)}')
                for metric in metrics:
                      print(f'Train {metric}: {round(train_metric_vals[metric][-1], 5)}\t', end='')
                print()
                print('-' * 40)
                
                if test_m:
                    print(f'Test losses: {[round(l, 5) for l in test_losses[-1]]}')
                    print(f'AVG test loss: {round(np.mean(test_losses[-1]), 5)}')
                    for metric in metrics:
                          print(f'Test {metric}: {round(test_metric_vals[metric][-1], 5)}\t', end='')
                    print()
                    print('-' * 40)

                print(f'Time: {round(time.time() - start, 3)}')
                print('=' * 40)

                #GRAPHICS
                fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(18, 18))

                colors = ['blue', 'orange', 'red', 'green', 'black', 'purple']
                DISPLAY_LAG = 30
                FIRST_IDX = max(1, epoch+1-DISPLAY_LAG)
                ticks = list(range(1, epoch+2))[FIRST_IDX:]
                
                train_losses_min = list(map(np.min, train_losses))[FIRST_IDX:]
                train_losses_mean = list(map(np.mean, train_losses))[FIRST_IDX:]
                train_losses_max = list(map(np.max, train_losses))[FIRST_IDX:]
                ax[0].plot(ticks, train_losses_min, c='b', linestyle='--')
                ax[0].plot(ticks, train_losses_mean, c='b', label='Train')
                ax[0].plot(ticks, train_losses_max, c='b', linestyle='--')
                
                if test_m:
                    test_losses_min = list(map(np.min, test_losses))[FIRST_IDX:]
                    test_losses_mean = list(map(np.mean, test_losses))[FIRST_IDX:]
                    test_losses_max = list(map(np.max, test_losses))[FIRST_IDX:]
                    ax[0].plot(ticks, test_losses_min, c='orange', linestyle='--')
                    ax[0].plot(ticks, test_losses_mean, c='orange', label='Test')
                    ax[0].plot(ticks, test_losses_max, c='orange', linestyle='--')
                
                if not test_m:
                    y_min = min(train_losses_min)
                    y_max = max(train_losses_max)
                else:
                    y_min = min(train_losses_min+test_losses_min)
                    y_max = max(train_losses_max+test_losses_max)
                y_min -= 0.3 * np.abs(y_min)
                y_max += 0.3 * np.abs(y_max)
                ax[0].set_ylim(y_min, y_max)
                ax[0].set_xticks(ticks)
                ax[0].set_xlabel('Epochs', fontsize=12)
                ax[0].set_ylabel('Loss', fontsize=12)
                ax[0].legend(loc=0, fontsize=12)
                ax[0].grid('on')
                
                for i, metric in enumerate(metrics):
                    ax[1].plot(ticks, train_metric_vals[metric][FIRST_IDX:], c=colors[i], label=f'Train {metric}')
                if test_m:
                    for i, metric in enumerate(metrics):
                        ax[1].plot(ticks, test_metric_vals[metric][FIRST_IDX:], c=colors[i], linestyle='--', label=f'Test {metric}')
                t = []
                for metric in metrics:
                    t += train_metric_vals[metric][FIRST_IDX:]
                if test_m:
                    for metric in metrics:
                        t += test_metric_vals[metric][FIRST_IDX:]
                y_min, y_max = min(t), max(t)
                y_min -= 0.3 * np.abs(y_min)
                y_max += 0.3 * np.abs(y_max)
                ax[1].set_ylim(y_min, y_max)
                ax[1].set_xticks(ticks)
                ax[1].set_xlabel('Epochs', fontsize=12)
                ax[1].set_ylabel('Metric', fontsize=12)
                ax[1].legend(loc=0, fontsize=12)
                ax[1].grid('on')
                
                ax[2].hist(epoch_true, bins=50, label='true dist', alpha=0.5)
                ax[2].hist(mode, bins=50, label='pred dist', alpha=0.5)
                ax[2].set_xlabel('Z', fontsize=12)
                ax[2].legend(loc=0, fontsize=12)
                ax[2].grid('on')
                
                plt.show()
                
        return train_metric_vals, train_losses, test_metric_vals, test_losses
    
# Метод для осуществления прогнозов.
# X - np.array с признаками объектов.
# batch_size - размер батча.
# sample_num - число сэмплов имперического распределения.
    def predict(self, X, batch_size=2**13, samples_num=288, save_gm_params=True):  
        X_tensor = torch.tensor(X, device=torch.device(self.device), dtype=torch.float)
        y_tensor = torch.tensor([0] * len(X_tensor), device=torch.device(self.device), dtype=torch.float)
        
        dataloader = self.BaseModel.get_dataloader(
            X_tensor, y_tensor,
            batch_size=batch_size, shuffle=False
        )
        
        epoch_pi, epoch_mu, epoch_sigma = [], [], []
        for i, model in enumerate(self.models):
            epoch_true, pi, mu, sigma = model.run_epoch(dataloader)
            epoch_pi.append(pi)
            epoch_mu.append(mu)
            epoch_sigma.append(sigma)
        epoch_pi = np.concatenate(epoch_pi, axis=1) / len(self.models)
        epoch_mu = np.concatenate(epoch_mu, axis=1)
        epoch_sigma = np.concatenate(epoch_sigma, axis=1)
        epoch_p = (1 / (epoch_sigma * np.sqrt(2 * np.pi))) * epoch_pi
        mode = self.mode_in_mus(epoch_pi, epoch_mu, epoch_sigma)
        mode_01 = epoch_mu[np.arange(epoch_mu.shape[0]), np.argmax(epoch_p, axis=1)]
        mu = np.sum(epoch_mu * epoch_pi, axis=1)
        sigma_a = np.sum((epoch_sigma ** 2) * epoch_pi, axis=1)
        sigma_e = np.sum((epoch_mu - mu.reshape(-1, 1))**2 * epoch_pi, axis=1)
        sigma = sigma_a + sigma_e
        
        result = pd.DataFrame()
        gauss_num = epoch_pi.shape[1]
        zfill_l = int(np.log10(gauss_num)) + 1
        # print(save_gm_params)
        if (save_gm_params != 0):
            for i in range(gauss_num):
                result[f'pi_{str(i).zfill(zfill_l)}'] = epoch_pi[:, i]
            for i in range(gauss_num):
                result[f'mu_{str(i).zfill(zfill_l)}'] = epoch_mu[:, i]
            for i in range(gauss_num):
                result[f'sigma_{str(i).zfill(zfill_l)}'] = epoch_sigma[:, i]
        result['mode'] = mode
        # result['mode_01'] = mode_01
        result['mu'] = mu
        result['sigma'] = sigma
        result['sigma_a'] = sigma_a
        result['sigma_e'] = sigma_e
        
        # zfill_l = int(np.log10(len(self.models))) + 1
        # mus_m = []
        # sigmas_m = []
        # for i in range(len(self.models)):
        #     i1 = i*self.models[0].n_gauss
        #     i2 = (i+1)*self.models[0].n_gauss
        #     mu_m = np.sum(epoch_mu[:,i1:i2] * epoch_pi[:,i1:i2]*len(self.models), axis=1)
        #     mus_m.append(mu_m.reshape(-1, 1))
        #     sigma_a_m = np.sum((epoch_sigma[:,i1:i2] ** 2) * epoch_pi[:,i1:i2]*len(self.models), axis=1)
        #     sigma_e_m = np.sum((epoch_mu[:,i1:i2] - mu_m.reshape(-1, 1))**2 * epoch_pi[:,i1:i2]*len(self.models), axis=1)
        #     sigma_m = sigma_a_m + sigma_e_m
        #     sigmas_m.append(sigma_m.reshape(-1, 1))
        #     result[f'mu_m_{str(i).zfill(zfill_l)}'] = mu_m
        #     result[f'sigma_m_{str(i).zfill(zfill_l)}'] = sigma_m
        #     result[f'sigma_a_m_{str(i).zfill(zfill_l)}'] = sigma_a_m
        #     result[f'sigma_e_m_{str(i).zfill(zfill_l)}'] = sigma_e_m
        # mus_m = np.hstack(mus_m)
        # print(mus_m.shape)
        # sigmas_m = np.hstack(sigmas_m)
        # print(sigmas_m.shape)
        # sigma_a_01 = np.mean((sigmas_m ** 2), axis=1)
        # sigma_e_01 = np.mean((mus_m - mu.reshape(-1, 1))**2, axis=1)
        # sigma_01 = sigma_a_01 + sigma_e_01
        # result['sigma_01'] = sigma_01
        # result['sigma_a_01'] = sigma_a_01
        # result['sigma_e_01'] = sigma_e_01
        
        # print(samples_num)
        if (samples_num is None) or (samples_num <= 0):
            return result
        
        preds_num = mode.shape[0]
        idx_01 = np.array([[i] * samples_num for i in range(preds_num)])
        idx_02 = []
        for i in range(mode.shape[0]):
            idx_02.append(np.random.choice(a=np.arange(0, gauss_num, 1, dtype=int), p=epoch_pi[i], size=samples_num))
            
        idx_02 = np.vstack(idx_02)
        samples_mu = epoch_mu[idx_01, idx_02]
        samples_sigma = epoch_sigma[idx_01, idx_02]
        samples = np.random.normal(samples_mu, samples_sigma)
        samples_cols = []
        zfill_l = int(np.log10(samples_num)) + 1
        for i in range(samples_num):
            samples_cols.append(f'sample_{str(i).zfill(zfill_l)}')
        samples_df = pd.DataFrame(samples, columns=samples_cols)
        result = pd.concat([result, samples_df], axis=1)
        
        result['zConf'] = (np.abs((samples - mode.reshape(-1, 1)) / (1 + mode.reshape(-1, 1))) < 0.06).sum(axis=1) / samples_num
        
        return result
    
    def mode_in_mus(self, pi, mu, sigma):
        def mode_in_point(pi, mu, sigma, point):
            return (pi * (1/(sigma*(2*np.pi)**0.5)) * np.exp(-0.5 * ((point - mu) / sigma) ** 2)).sum(1).reshape(-1, 1)
        p_in_mus = []
        for i in range(pi.shape[1]):
            p_in_mus.append(mode_in_point(pi, mu, sigma, mu[:,i].reshape(-1, 1)))
        p_in_mus = np.hstack(p_in_mus)
        return mu[np.arange(pi.shape[0]), p_in_mus.argmax(1)]
    
    