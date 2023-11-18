import numpy as np
import pandas as pd
from random import choice
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

class DataProcessor:
    def __init__(self, n_bin=3):
        self.n_bin = n_bin

    def process(self, X, prob, cut_method='cut'):
        """
        Process the data based on the problem type
        :param X: Data X
        :param prob: Problem name
        :param cut_method: Method for data cut
        :return: Processed data
        """
        cut_method = eval('pd.' + cut_method)

        if prob == 'spam':
            X = self._process_spam(X, cut_method)
        elif prob == 'volcanoes':
            X = self._process_volcanoes(X, cut_method)
        elif prob == 'voting':
            X = self._process_voting(X)

        return X

    def _process_spam(self, X, cut_method):
        """
        Process data for the 'spam' problem
        :param X: Data X
        :param cut_method: Method for data cut
        :return: Processed data
        """
        for i in range(X.shape[1]):
            if i != 6:  # Exclude the 'Windows' column
                X[:, i] = pd.to_numeric(X[:, i], errors='coerce')

        # Removing NaN values and the 'Windows' column
        df = pd.DataFrame(X)
        df = df.drop(df.columns[5], axis=1)
        X = df.values

        # Apply the cut_method to numeric columns
        for i in range(X.shape[1]):
            if np.issubdtype(X[:, i].dtype, np.number):
                X[:, i] = cut_method(X[:, i], self.n_bin, retbins=True, labels=[x for x in range(self.n_bin)])[0].codes

        X = X.astype(np.float32)
        return X

    def _process_volcanoes(self, X, cut_method):
        """
        Process data for the 'volcanoes' problem
        :param X: Data X
        :param cut_method: Method for data cut
        :return: Processed data
        """
        X = X.astype(np.int32)
        X = X[:, 1:]  # Remove the first column

        # Apply the cut_method to all columns
        for i in range(X.shape[1]):
            X[:, i] = cut_method(X[:, i], self.n_bin, retbins=True, labels=[x for x in range(self.n_bin)])[0].codes

        return X

    def _process_voting(self, X):
        """
        Process data for the 'voting' problem
        :param X: Data X
        :return: Processed data
        """
        for i in range(X.shape[1]):
            # Check if the column has non-numeric values
            if not np.issubdtype(np.array(X[:, i]).dtype, np.number):
                X[:, i] = pd.Categorical(X[:, i]).codes

        X = X.astype(np.int32)
        return X


class ReportGenerator:
    @staticmethod
    def report_cross(acc, prec, rec):
        """
        Generate a cross-validation report
        :param acc: Accuracy
        :param prec: Precision
        :param rec: Recall
        """
        print("===============Fold report==================")
        print('Accuracy:{:.03f}'.format(acc))
        print('Precision:{:.03f}'.format(prec))
        print('Recall:{:.03f}'.format(rec))

    @staticmethod
    def report_final(acc, prec, rec, auc):
        """
        Generate a final report
        :param acc: Accuracy
        :param prec: Precision
        :param rec: Recall
        :param auc: Area under ROC
        """
        print("===============Final report=================")
        if type(acc) is list:
            print('Accuracy:{:.03f} {:.03f}'.format(np.mean(acc), np.std(acc)))
            print('Precision:{:.03f} {:.03f}'.format(np.mean(prec), np.std(prec)))
            print('Recall:{:.03f} {:.03f}'.format(np.mean(rec), np.std(rec)))
            print('Area under ROC {:.03f}'.format(auc))
        else:
            print('Accuracy:{:.03f} {:.03f}'.format(acc, 0))
            print('Precision:{:.03f} {:.03f}'.format(prec, 0))
            print('Recall:{:.03f} {:.03f}'.format(rec, 0))
            print('Area under ROC {:.03f}'.format(auc))

class DataLoader:
    @staticmethod
    def n_fold(n_sample, n_fold=5):
        """
        Generate indices for n-fold cross-validation
        :param n_sample: Number of samples
        :param n_fold: Number of folds
        :return: List of fold indices
        """
        a = list(range(n_sample))
        folds = [[] for i in range(n_fold)]
        fold_ptr = 0
        while len(a) > 0:
            t = choice(a)
            a.remove(t)
            folds[fold_ptr].append(t)
            fold_ptr = (fold_ptr + 1) % n_fold

        return folds

    @staticmethod
    def read_data(path, n_bin=3):
        """
        Read and process data from a file
        :param path: File path
        :param n_bin: Number of bins
        :return: Processed data and labels
        """
        prob_name = path.split('/')[-1]
        datafile = path + '/' + prob_name + '.data'
        data = np.loadtxt(datafile, delimiter=',', dtype=str)
        X = data[:, 1:-1]
        X = DataProcessor(n_bin).process(X, prob_name)
        y = data[:, -1].astype(int)
        return X, y


class EvaluationMetrics:
    @staticmethod
    def compute_tp_tn_fn_fp(y, y_hat):
        """
        Compute true positives, true negatives, false positives, and false negatives
        :param y: True labels
        :param y_hat: Predicted labels
        :return: Tuple (tp, tn, fp, fn)
        """
        tp = sum((y == 1) & (y_hat == 1))
        tn = sum((y == 0) & (y_hat == 0))
        fn = sum((y == 1) & (y_hat == 0))
        fp = sum((y == 0) & (y_hat == 1))
        return tp, tn, fp, fn

    @staticmethod
    def compute_accuracy(tp, tn, fn, fp):
        """
        Compute accuracy
        :param tp: True positives
        :param tn: True negatives
        :param fn: False negatives
        :param fp: False positives
        :return: Accuracy
        """
        return (tp + tn) / float(tp + tn + fn + fp)

    @staticmethod
    def compute_precision(tp, fp):
        """
        Compute precision
        :param tp: True positives
        :param fp: False positives
        :return: Precision
        """
        return tp / float(tp + fp)

    @staticmethod
    def compute_recall(tp, fn):
        """
        Compute recall
        :param tp: True positives
        :param fn: False negatives
        :return: Recall
        """
        return tp / float(tp + fn)


def split_data(X, y, test_fold_idx, folds):
    train_idx = []
    for f in range(5):
        if f != test_fold_idx:
            train_idx.extend(folds[f])

    trainX = X[train_idx, :]
    trainy = y[train_idx]
    testX = X[folds[test_fold_idx], :]
    testy = y[folds[test_fold_idx]]

    return trainX, trainy, testX, testy


class PerformanceEvaluator:
    @staticmethod
    def cal_APR(y_hat, y):
        """
        Compute accuracy, precision, and recall
        :param y_hat: Predicted labels
        :param y: True labels
        :return: Tuple (accuracy, precision, recall)
        """
        tp, tn, fp, fn = EvaluationMetrics.compute_tp_tn_fn_fp(y, y_hat)
        acc = EvaluationMetrics.compute_accuracy(tp, tn, fn, fp)
        prec = EvaluationMetrics.compute_precision(tp, fp)
        rec = EvaluationMetrics.compute_recall(tp, fn)
        return acc, prec, rec

    @staticmethod
    def cal_LR_APR(pred, y):
        """
        Compute accuracy, precision, and recall for Logistic Regression predictions
        :param pred: Predicted probabilities
        :param y: True labels
        :return: Tuple (accuracy, precision, recall)
        """
        n = len(y)
        y_hat = pred > 0.5
        return PerformanceEvaluator.cal_APR(y_hat, y)

    @staticmethod
    def cal_bayes_APR(pred_res, y):
        """
        Compute accuracy, precision, and recall for Bayesian predictions
        :param pred_res: Predicted probabilities
        :param y: True labels
        :return: Tuple (accuracy, precision, recall)
        """
        n = len(y)
        nega = pred_res[:, 0]
        posi = pred_res[:, 1]
        y_hat = posi > nega
        return PerformanceEvaluator.cal_APR(y_hat, y)

    @staticmethod
    def cal_AUC(y, y_hat, num_bins=10000):
        """
        Compute the Area Under the Curve (AUC) for ROC
        :param y: True labels
        :param y_hat: Predicted probabilities
        :param num_bins: Number of bins for histogram
        :return: AUC score
        """
        postive_len = sum(y)
        negative_len = len(y) - postive_len
        total_grid = postive_len * negative_len
        pos_histogram = [0 for _ in range(num_bins + 1)]
        neg_histogram = [0 for _ in range(num_bins + 1)]
        bin_width = 1.0 / num_bins

        for i in range(len(y)):
            nth_bin = int(y_hat[i] / bin_width)
            pos_histogram[nth_bin] += 1 if y[i] == 1 else 0
            neg_histogram[nth_bin] += 1 if y[i] == 0 else 0

        accu_neg = 0
        satisfied_pair = 0

        for i in range(num_bins):
            satisfied_pair += pos_histogram[i] * accu_neg + pos_histogram[i] * neg_histogram[i] * 0.5
            accu_neg += neg_histogram[i]

        return satisfied_pair / float(total_grid)

    @staticmethod
    def plot_roc(y_true, y_score , filename):
        """
        Plot the Receiver Operating Characteristic (ROC) curve
        :param y_true: True labels
        :param y_score: Predicted probabilities
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        pos_label = 1.

        # Make y_true a boolean vector
        y_true = (y_true == pos_label)

        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        out = np.cumsum(y_true, dtype=np.float64)
        expected = np.sum(y_true, dtype=np.float64)
        tps = out[threshold_idxs]

        fps = 1 + threshold_idxs - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        fpr = fps / fps[-1]
        tpr = tps / tps[-1]

        
        plt.figure()
        plt.title('Receiver Operating Characteristic')
        df_plot = pd.DataFrame({'fpr':fpr , 'tpr':tpr})
        fig = px.line(df_plot , x = 'fpr' , y = 'tpr')
        fig.write_image(f'{filename}.png')
        # plt.plot(fpr,tpr)
        # # plt.legend(loc='lower right')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
