import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import pickle
import pandas as pd
from scipy.io.arff import loadarff
from xclib.data import data_utils
import math


def p_k(y_predicted, y_actual, k):
    res = 0
    for i in range(len(y_predicted)):
        indices = y_predicted[i].argsort()[-k:][::-1]
        res += (y_actual[i][indices].sum())/k
    return res/len(y_predicted)


def dcg_k(y_predicted, y_actual, k):
    def logger(t): return 1/math.log(1+t, 2)
    logger = np.vectorize(logger)
    # l = np.arange(1,1+y_predicted.shape[1])
    l = np.arange(1, 1+k)
    l = logger(l)
    res = np.zeros(y_predicted.shape[0])
    for i in range(len(y_predicted)):
        indices = y_predicted[i].argsort()[-k:][::-1]
        temp = l*y_actual[i][indices]
        res[i] = temp.sum()
        # temp = l*y_actual[i]
        # res[i] = temp[indices].sum()
    return res


def n_k(y_predicted, y_actual, k):
    dcg_k_list = dcg_k(y_predicted, y_actual, k)
    def logger(t): return 1/math.log(1+t, 2)
    logger = np.vectorize(logger)
    res = 0
    for i in range(len(y_predicted)):
        lim = y_actual[i].sum()
        if(lim == 0):
            continue
        l = np.arange(1, 1+min(lim, k))
        l = logger(l)
        deno = l.sum()
        res += dcg_k_list[i]/deno
    return res/y_predicted.shape[0]


def print_scores(y_pred, Y_test):
    print("#########################")
    print("P@1: ", p_k(y_pred, Y_test, 1))
    print("P@3: ", p_k(y_pred, Y_test, 3))
    print("P@5: ", p_k(y_pred, Y_test, 5))
    print("#########################")
    print("n@1: ", n_k(y_pred, Y_test, 1))
    print("n@3: ", n_k(y_pred, Y_test, 3))
    print("n@5: ", n_k(y_pred, Y_test, 5))


def get_matrix_from_txt(path):
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        path)
    return features.toarray(), labels.toarray().astype('int')


def load_rcv_data(path):
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        path)
    return features, labels


def get_data(path):
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    X_cols = df_data.columns[:120]
    Y_cols = df_data.columns[120:]
    X_data = df_data[X_cols]
    Y_data = df_data[Y_cols]
    for col in Y_data.columns:
        Y_data[col] = Y_data[col].apply(lambda x: int(x.decode("utf-8")))
        # Y_data[col] = Y_data[col].astype(int)
    return X_data.values, Y_data.values


def load_small_data(full_data_path, tr_path, tst_path):
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        full_data_path)
    labels = labels.toarray()
    features = features.toarray()

    dum_feature = np.zeros((1, num_features))
    dum_label = np.zeros((1, num_labels))
    features = np.concatenate((dum_feature, features), axis=0)
    labels = np.concatenate((dum_label, labels), axis=0)

    train_indices = pd.read_csv(tr_path, sep=" ", header=None)
    tr_indices = train_indices.to_numpy()
    train_X = features[tr_indices[:, 0]]
    train_Y = labels[tr_indices[:, 0]]

    test_indices = pd.read_csv(tst_path, sep=" ", header=None)
    tst_indices = test_indices.to_numpy()
    test_X = features[tst_indices[:, 0]]
    test_Y = labels[tst_indices[:, 0]]
    return train_X, train_Y, test_X, test_Y


def load_data(dataset_name, full_path, train_path, test_path):
    if(dataset_name in ['DELICIOUS', 'MEDIAMILL']):
        X_train, Y_train, X_test, Y_test = load_small_data(
            full_data_path=full_path,
            tr_path=train_path,
            tst_path=test_path
        )
    if(dataset_name in ['EURLEX', 'RCV']):
        X_train, Y_train = get_matrix_from_txt(path=train_path)
        X_test, Y_test = get_matrix_from_txt(path=test_path)
    return X_train, Y_train, X_test, Y_test


def load_embeddings(embedding_path):
    if(embedding_path is None):
        return None
    embedding_weights = np.loadtxt(
        open(embedding_path, "rb"), delimiter=",", skiprows=0)
    embedding_weights = torch.from_numpy(embedding_weights)
    return embedding_weights


def split_train_val(X, tfidf, Y):
    indices = np.random.permutation(X.shape[0])
    val_size = int(0.2*X.shape[0])
    val_idx, test_idx = indices[:val_size], indices[val_size:]
    X_val = X[val_idx]
    tfidf_val = tfidf[val_idx]
    Y_val = Y[val_idx]
    X_test = X[test_idx]
    tfidf_test = tfidf[test_idx]
    Y_test = Y[test_idx]
    return X_val, tfidf_val, Y_val, X_test, tfidf_test, Y_test


def prepare_tensors_from_data(X_train, Y_train):
    X_data_new = np.zeros(X_train.shape)
    non_zero_indexes = np.nonzero(X_train)
    X_data_new[non_zero_indexes] = 1
    X_TfIdftensor = torch.from_numpy(X_train[:, :, None])
    X_train = torch.from_numpy(X_data_new)
    Y_train = torch.from_numpy(Y_train)
    X_train = X_train.type('torch.LongTensor')
    Y_train = Y_train.type('torch.FloatTensor')
    X_TfIdftensor = X_TfIdftensor.type('torch.FloatTensor')
    print(X_train.shape, X_TfIdftensor.shape, Y_train.shape)
    return X_train, X_TfIdftensor, Y_train


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def to_numpy(tensor):
    return tensor.cpu().numpy()


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
