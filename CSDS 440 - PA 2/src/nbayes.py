import sys
import numpy as np
from collections import Counter
from math import log
from util import PerformanceEvaluator, ReportGenerator, DataLoader, split_data
# from util import read_data, cal_bayes_APR, cal_AUC, report_cross, report, n_fold

class NaiveBayesClassifier:
    def __init__(self, op4):
        self.op4 = op4
        self.Laplace = True if op4 < 0 else False
        self.pre_p = None
        self.posi_p = None
        self.nega_p = None

    @staticmethod
    def _pred(x, pre_p, posi_p, nega_p):
        res = [log(pre_p[0]), log(pre_p[1])]  # negative / positive
        for i, attr in enumerate(x):
            res[1] += log(posi_p[i][attr])
            res[0] += log(nega_p[i][attr])

        return res

    def train(self, X, y, posi_num, nega_num):
        m = len(y)
        n = len(X[0])

        class_num = Counter(y.reshape(-1))
        self.pre_p = {k: v / m for k, v in class_num.items()}

        self.posi_p = [{} for _ in range(n)]
        self.nega_p = [{} for _ in range(n)]

        for i, d in enumerate(posi_num):
            _posi_num_i = Counter(X[y.reshape(-1) == 1, i])
            for attr in _posi_num_i:
                posi_num[i][attr] = _posi_num_i[attr]

        for i, d in enumerate(nega_num):
            _nega_num_i = Counter(X[y.reshape(-1) == 0, i])
            for attr in _nega_num_i:
                nega_num[i][attr] = _nega_num_i[attr]

        for i, d in enumerate(self.posi_p):
            num_value_in_attr = len(posi_num[i])
            p = 1 / num_value_in_attr
            m = num_value_in_attr if self.Laplace else self.op4
            for attr in posi_num[i]:
                d[attr] = (posi_num[i][attr] + m * p) / (sum(y == 1) + m)

        for i, d in enumerate(self.nega_p):
            num_value_in_attr = len(nega_num[i])
            p = 1 / num_value_in_attr
            m = num_value_in_attr if self.Laplace else self.op4
            for attr in nega_num[i]:
                d[attr] = (nega_num[i][attr] + m * p) / (sum(y == 0) + m)

    def predict(self, X):
        if self.pre_p is not None and self.posi_p is not None and self.nega_p is not None:
            ret = []
            for x in X:
                ret.append(self._pred(x, self.pre_p, self.posi_p, self.nega_p))

            return np.asarray(ret)
        else:
            raise ValueError("Model is not trained yet.")
def main():
    argv = sys.argv[1:]
    path = argv[0]
    n_bin = int(argv[2])
    X, y = DataLoader.read_data(path, n_bin=n_bin)

    n = len(X[0])
    posi_num = [{} for _ in range(n)]
    nega_num = [{} for _ in range(n)]

    for i, d in enumerate(posi_num):
        for attr in np.unique(X[:, i]):
            posi_num[i][attr] = 0

    for i, d in enumerate(nega_num):
        for attr in np.unique(X[:, i]):
            nega_num[i][attr] = 0

    cross_validation = argv[1] == '1'
    if cross_validation:
        folds = DataLoader.n_fold(len(X))
        AUC_y = []
        pred_AUC_y = []
        pred_AUC_y_nega = []
        acc = []
        prec = []
        rec = []
        for test_fold_idx in range(5):
            trainX, trainy, testX, testy = split_data(X, y, test_fold_idx, folds)
            
            model = NaiveBayesClassifier(float(argv[3]))
            model.train(trainX, trainy, posi_num, nega_num)
            pred_res = model.predict(testX)

            AUC_y.extend(testy)
            pred_AUC_y.extend(np.exp(pred_res[:, 1]))
            pred_AUC_y_nega.extend(np.exp(pred_res[:, 0]))

            _acc, _prec, _rec = PerformanceEvaluator.cal_bayes_APR(pred_res, testy)
            ReportGenerator.report_cross(_acc, _prec, _rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)

        roc_score = PerformanceEvaluator.cal_AUC(AUC_y, pred_AUC_y)
        pred_AUC_y = np.asarray(pred_AUC_y)
        pred_AUC_y_nega = np.asarray(pred_AUC_y_nega)
        ReportGenerator.report_final(acc, prec, rec, roc_score)
    else:
        model = NaiveBayesClassifier(float(argv[3]))
        model.train(X, y, posi_num, nega_num)
        pred_res = model.predict(X)

        roc_score = PerformanceEvaluator.cal_AUC(y, np.exp(pred_res[:, 1]))

        acc, prec, rec = PerformanceEvaluator.cal_bayes_APR(pred_res, y)
        ReportGenerator.report_final(acc, prec, rec, roc_score)

if __name__ == '__main__':
    main()
