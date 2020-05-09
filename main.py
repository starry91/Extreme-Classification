import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from utils import *
import time
import logging
import gzip
import os
import math
from scipy.io.arff import loadarff
from torchviz import make_dot
import numpy as np
from torch import optim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torchsummary import summary
from torch import nn
import torch
from model.net import AttentionModel
from losses import Loss
import sys
from configparser import ConfigParser, ExtendedInterpolation
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
torch.manual_seed(1)

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

torch.set_default_tensor_type(torch.FloatTensor)


class Solver():
    def __init__(self, model, loss, outdim_size, params, lamda=50, device=torch.device('cpu')):
        self.model = model
        self.model.to(device)
        self.epoch_num = params['epoch_num']
        self.batch_size = params['batch_size']
        self.loss = loss
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=params['learning_rate'], weight_decay=params['reg_par'])
        self.device = device

        self.reg_par = params['reg_par']
        self.outdim_size = outdim_size
        self.lamda = lamda

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("XML.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)
        self.start_epoch = 0

    def fit(self, X_train, tfidf, Y_train,
            v_x=None, v_tfidf=None, v_y=None,
            t_x=None, t_tfidf=None, t_y=None,
            checkpoint='', load_checkpoint=False):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        X_train = X_train
        Y_train = Y_train
        tfidf = tfidf
        data_size = X_train.size(0)
        path_strings = checkpoint.split('/')
        path_strings[-1] = "train_"+path_strings[-1]
        train_check_path = '/'.join(path_strings)
        if v_x is not None and v_y is not None:
            best_val_loss = None
            v_x
            v_tfidf
            v_y
        if t_x is not None and t_y is not None:
            t_x
            t_tfidf
            t_y

        if(load_checkpoint):
            self.start_epoch, loss = self.load_model(checkpoint)

        train_losses = []
        train_loss_hidden_list = []
        train_loss_ae_list = []
        val_losses = []
        val_loss_hidden_list = []
        val_loss_ae_list = []
        while(self.start_epoch < self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))

            temp_loss = []
            temp_loss_hidden = []
            temp_loss_ae = []
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_tfidf = tfidf[batch_idx].to(self.device)
                batch_X_train = X_train[batch_idx, :].to(self.device)
                batch_Y_train = Y_train[batch_idx, :].to(self.device)
                x_hidden, y_hidden, y_predicted = self.model(
                    batch_X_train, batch_tfidf, batch_Y_train)
                loss_hidden, loss_ae = self.loss(x_hidden, y_hidden,
                                                 y_predicted, batch_Y_train)
                loss = loss_hidden+self.lamda*loss_ae
                temp_loss.append(loss.item())
                temp_loss_hidden.append(loss_hidden.item())
                temp_loss_ae.append(loss_ae.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(np.array(temp_loss))
            train_losses.append(train_loss)
            train_loss_hidden_list.append(np.mean(np.array(temp_loss_hidden)))
            train_loss_ae_list.append(np.mean(np.array(temp_loss_ae)))

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if v_x is not None and v_y is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss, val_hidden, val_ae = self.test(
                        v_x, v_tfidf, v_y)
                    val_losses.append(val_loss)
                    val_loss_hidden_list.append(
                        np.mean(np.array(val_hidden)))
                    val_loss_ae_list.append(
                        np.mean(np.array(val_ae)))
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if(best_val_loss is None):
                        best_val_loss = val_loss
                    elif val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(self.start_epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': self.start_epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                        }, checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            self.start_epoch + 1, best_val_loss))
            torch.save({
                'epoch': self.start_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, train_check_path)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                self.start_epoch + 1, self.epoch_num, epoch_time, train_loss))
            self.start_epoch += 1

            self.save_loss_plots(np.array(train_losses),
                                 np.array(train_loss_hidden_list),
                                 np.array(train_loss_ae_list),
                                 np.array(val_losses),
                                 np.array(val_loss_hidden_list),
                                 np.array(val_loss_ae_list),
                                 self.start_epoch)

        checkpoint_ = torch.load(checkpoint)['model_state_dict']
        self.model.load_state_dict(checkpoint_)
        if v_x is not None and v_y is not None:
            loss, _, _ = self.test(v_x, v_tfidf, v_y)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if t_x is not None and t_y is not None:
            loss, _, _ = self.test(t_x, t_tfidf, t_y)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def save_loss_plots(self, train_losses, train_hidden, train_ae,
                        val_losses, val_hidden, val_ae, epoch):
        if(epoch <= 50):
            fig_name = "losses_50_{0}.png".format(self.lamda)
        else:
            fig_name = "losses_{0}.png".format(self.lamda)
        fig, ax = plt.subplots(2, 3)
        fig.set_size_inches((30, 18))
        fig.suptitle("losses_{0}".format(self.lamda), fontsize=20)
        ax[0][0] = plt.subplot(231)

        ax[0][0].set_title("train_loss")
        ax[0][0].plot(train_losses, 'r')
        ax[0][1].set_title("train_hidden")
        ax[0][1].plot(train_hidden, 'r')
        ax[0][2].set_title("train_reconstruction")
        ax[0][2].plot(train_ae, 'r')

        ax[1][0].set_title("val_loss")
        ax[1][0].plot(val_losses, 'r')
        ax[1][1].set_title("val_hidden")
        ax[1][1].plot(val_hidden, 'r')
        ax[1][2].set_title("val_reconstruction")
        ax[1][2].plot(val_ae, 'r')
        fig.savefig(fig_name)

    def test(self, x, tfidf, y):
        x = x
        tfidf = tfidf
        y = y
        with torch.no_grad():
            self.model.eval()
            data_size = x.shape[0]
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            loss_hidden_list = []
            loss_ae_list = []
            for batch_idx in batch_idxs:
                batch_x1 = x[batch_idx, :].to(self.device)
                batch_tfidf = tfidf[batch_idx].to(self.device)
                batch_y = y[batch_idx].to(self.device)
                x_hidden, y_hidden, y_predicted = self.model(
                    batch_x1, batch_tfidf, batch_y)
                loss_hidden, loss_ae = self.loss(x_hidden, y_hidden,
                                                 y_predicted, batch_y)
                loss = loss_hidden+self.lamda*loss_ae
                losses.append(loss.item())
                loss_hidden_list.append(loss_hidden.item())
                loss_ae_list.append(loss_ae.item())
        return np.mean(losses), np.mean(loss_hidden_list), np.mean(loss_ae_list)

    def load_model(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint_ = torch.load(path)
        self.model.load_state_dict(checkpoint_['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_['optimizer_state_dict'])
        start_epoch = int(checkpoint_['epoch'])+1
        loss = checkpoint_['loss']
        return start_epoch, loss

    def predict(self, X_test, tfidf):
        tfidf = tfidf
        bow = X_test
        with torch.no_grad():
            self.model.eval()
            data_size = X_test.shape[0]
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            outputs1 = []
            for batch_idx in batch_idxs:
                batch_x1 = bow[batch_idx, :].to(device)
                batch_tfidf = tfidf[batch_idx].to(device)
                o1 = self.model.predict(batch_x1, batch_tfidf)
                outputs1.append(o1)
        outputs = torch.cat(outputs1, dim=0)
        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)
    print("Using", torch.cuda.device_count(), "GPUs")

    config = ConfigParser(
        allow_no_value=True, interpolation=ExtendedInterpolation())
    config.read('params.config')
    dataset = config['COMMON']['dataset']
    dataset_conf = config[dataset]
    # the size of the new space learned by the model (number of the new features)
    input_size = int(dataset_conf['input_size'])
    output_size = int(dataset_conf['output_size'])
    embedding_size = int(dataset_conf['embedding_size'])
    attention_layer_size = int(dataset_conf['attention_layer_size'])
    encoder_layer_size = int(dataset_conf['encoder_layer_size'])
    hidden_layer_size = int(dataset_conf['hidden_layer_size'])

    # the parameters for training the network
    params = dict()
    params['learning_rate'] = float(dataset_conf['learning_rate'])
    params['epoch_num'] = int(dataset_conf['epoch_num'])
    params['batch_size'] = int(dataset_conf['batch_size'])
    params['reg_par'] = float(dataset_conf['reg_par'])

    r1 = float(dataset_conf['r1'])
    m = float(dataset_conf['m'])
    lamda = int(dataset_conf['lamda'])
    use_all_singular_values = dataset_conf.getboolean(
        'use_all_singular_values')

    X_train, Y_train, X_test, Y_test = load_data(dataset,
                                                 dataset_conf['full_path'],
                                                 dataset_conf['train_path'],
                                                 dataset_conf['test_path'])

    is_inference = dataset_conf.getboolean('is_inference')
    is_training_inference = dataset_conf.getboolean('is_training_inference')
    embedding_weights = load_embeddings(dataset_conf['embedding_path'])
    check_path = dataset_conf['check_path']

    ### Common code from here #########
    X_train, train_tfidf, Y_train = prepare_tensors_from_data(X_train, Y_train)
    X_test, test_tfidf, Y_test = prepare_tensors_from_data(X_test, Y_test)
    X_val, tfidf_val, Y_val, _, _, _ = split_train_val(
        X_test, test_tfidf, Y_test)
    # Building, training, and producing the new features by DCCA
    model = AttentionModel(input_size=input_size, embedding_size=embedding_size,
                           attention_layer_size=attention_layer_size, encoder_layer_size=encoder_layer_size,
                           hidden_layer_size=hidden_layer_size, output_size=output_size,
                           embedding_weight_matrix=embedding_weights).to(device)
    loss_func = Loss(outdim_size=hidden_layer_size, use_all_singular_values=use_all_singular_values,
                     device=device, r1=r1, m=m).loss
    solver = Solver(model=model, loss=loss_func,
                    outdim_size=output_size, params=params, lamda=lamda, device=device)

    if(is_inference and is_training_inference):
        path_strings = check_path.split('/')
        path_strings[-1] = "train_"+path_strings[-1]
        check_path = '/'.join(path_strings)

    if(not is_inference):
        solver.fit(X_train, train_tfidf, Y_train,
                   X_val, tfidf_val, Y_val,
                   checkpoint=check_path, load_checkpoint=False)
    else:
        solver.load_model(check_path)

    # Test data scores
    y_pred = solver.predict(X_test, test_tfidf)
    y_pred = to_numpy(y_pred)
    Y_act = to_numpy(Y_test)
    print("Test data scores")
    print_scores(y_pred, Y_act)

    # Train data scores
    y_pred = solver.predict(X_train, train_tfidf)
    y_pred = to_numpy(y_pred)
    Y_act = to_numpy(Y_train)
    print("Train data scores")
    print_scores(y_pred, Y_act)

    d = torch.load(check_path)
    solver.model.load_state_dict(d['model_state_dict'])
    solver.model.parameters()
