import sys
from random import seed
import numpy as np
from util import PerformanceEvaluator, ReportGenerator, DataLoader, split_data
import re

class LogisticRegression:
    def __init__(self, lbd=0.1, learning_rate=0.001, max_iters=500):
        self.lbd = lbd
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        X = np.mat(X)
        y = np.mat(y).transpose()
        m, n = np.shape(X)

        # Initialize weights
        self.weights = np.ones((n, 1))

        for i in range(self.max_iters):
            y_hat = self.sigmoid(X * self.weights)
            grad = X.transpose() * (y - y_hat) + self.lbd * self.weights
            self.weights = self.weights + self.learning_rate * grad

    def predict(self, X):
        if self.weights is not None:
            return np.asarray(self.sigmoid(X * self.weights)).reshape(-1)
        else:
            raise ValueError("Model is not trained yet")

def main():
    seed(12345)
    argv = sys.argv[1:]
    path = argv[0]
    X, y = DataLoader.read_data(path)
    lbd = float(argv[2])
    match = re.search(r'/([^/]+)$', path)
    fname = match.group(1) + '_' + re.sub(r'\.', '', str(lbd))
    cross_validation = argv[1] == '1'
    folds = DataLoader.n_fold(len(X))
    plot_roc = argv[3] == '1'
    if cross_validation:
        AUC_y = []
        pred_AUC_y = []
        acc = []
        prec = []
        rec = []
        for test_fold_idx in range(5):
            trainX, trainy, testX, testy = split_data(X, y, test_fold_idx, folds)

            model = LogisticRegression(lbd=lbd)
            model.train(trainX, trainy)
            pred_res = model.predict(testX)

            AUC_y.extend(testy)
            pred_AUC_y.extend(pred_res)

            _acc, _prec, _rec = PerformanceEvaluator.cal_LR_APR(pred_res, testy)
            ReportGenerator.report_cross(_acc, _prec, _rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)

        roc_score = PerformanceEvaluator.cal_AUC(AUC_y, pred_AUC_y)
        ReportGenerator.report_final(acc, prec, rec, roc_score)
        if plot_roc:
            PerformanceEvaluator.plot_roc(testy, pred_res , fname)
    else:
        model = LogisticRegression(lbd=lbd)
        model.train(X, y)
        pred_res = model.predict(X)
        acc, prec, rec = PerformanceEvaluator.cal_LR_APR(pred_res, y)
        roc_score = PerformanceEvaluator.cal_AUC(y, pred_res)
        ReportGenerator.report_final(acc, prec, rec, roc_score)
        if plot_roc:
            PerformanceEvaluator.plot_roc(testy, pred_res , fname)

if __name__ == '__main__':
    main()
