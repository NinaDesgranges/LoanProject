import pandas as pd
from settings import DATA, DATA_LOCAL
from sklearn import preprocessing
import matplotlib.pyplot as plt
import collections
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import base
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, f1_score, recall_score, accuracy_score, precision_score
import json
from pprint import pprint
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import Figure
from bokeh.io import show
class dataTransform(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, columns, applyTransformation, columnsAppend):
        self.columns = columns
        self.applyTransformation = applyTransformation
        self.columnsAppend = columnsAppend

    def fit(self, X):
        self.applyTransformation.fit(X[self.columns].to_dict(orient='records'))
        print X[self.columns].to_dict(orient='records')
        self.feature_names = self.applyTransformation.get_feature_names()
        print self.feature_names
        return self

    def transform(self, X):
        matrix = self.applyTransformation.transform(X[self.columns].to_dict(orient='records'))
        for column in self.columnsAppend:
            matrix = np.concatenate((matrix, X[column].values.reshape(1, -1).T), axis=1)
        columns_name = self.feature_names + self.columnsAppend
        matrix = pd.DataFrame(matrix, columns=columns_name).round(3)
        to_int = self.feature_names + ['loan']
        matrix[to_int] = matrix[to_int].astype(int)
        return matrix


def preliminaryAnalysis():
    data = pd.read_csv(DATA + "accepted_refused_ds.csv", header=0)

    print 'Data shape = ' + str(data.shape)
    print 'Columns = ' + ', '.join(data.columns.values)
    month = [c[0:3] for c in data.date]

    ct = collections.Counter(data.loan).items()
    print ct
    x_val = map(lambda x: str(x).replace('0', 'Refused').replace('1', 'Accepted'), [c[0] for c in ct])

    y_val = [c[1] for c in ct]
    x_val = [c[0] for c in ct]
    # [str(c).replace('0, 'Refused').replace(1, 'Accepted') for c in ct.keys()]

    plt.bar(x_val, y_val, width=1, color=['#00BF2D'] * 2)
    plt.title('Accepted and Refused Loans')
    plt.xticks([0.5, 1.5], ['Refused', 'Accepted'])
    plt.savefig(DATA + 'images/perc_accepted_refused.png')
    plt.close()

    # plt.boxplot(data.dti[data.dti != -1])
    plt.boxplot(data.dti)
    plt.title('Debt To Income Ratio')
    plt.ylim([-10, +100])
    plt.savefig(DATA + 'images/debt_to_income_zoom_v1.png')
    plt.close()

    # plt.boxplot(data.dti[data.dti != -1])
    plt.boxplot(data.dti)
    plt.title('Debt To Income Ratio')
    plt.ylim([-10, +200])
    plt.savefig(DATA + 'images/debt_to_income_zoom_v2.png')
    plt.close()

    # plt.boxplot(data.dti[data.dti != -1])
    plt.boxplot(data.dti)
    plt.title('Debt To Income Ratio')
    plt.ylim([-10, +1000])
    plt.savefig(DATA + 'images/debt_to_income_zoom_v3.png')
    plt.close()

    # plt.boxplot(data.dti[data.dti != -1])
    plt.boxplot(data.dti)
    plt.title('Debt To Income Ratio')
    plt.savefig(DATA + 'images/debt_to_income.png')
    plt.close()

    print data.dti.describe()


def dataTransformation(path=DATA_LOCAL + "accepted_refused_ds.csv"):
    print 'Open data'
    data = pd.read_csv(path, header=0)
    data['month'] = [c[0:3] for c in data.date]
    print data.columns.values

    print 'Data transformation'
    feat = dataTransform(columns=['month', 'emp_len', 'state'],
                         applyTransformation=DictVectorizer(sparse=False),
                         columnsAppend=['amnt', 'dti', 'loan'])

    print 'Fit the data transformation'

    data_transformed = feat.fit_transform(data)

    # data_transformed.to_csv(DATA + 'accepted_refused_ds_small_trans.csv', index=False)

    return data_transformed


def gridSearchLogisticRegression(data):
    print data.head()

    y = data['loan']
    X = data.drop('loan', axis=1)

    print X.head()

    lr = LogisticRegression(class_weight='balanced', random_state=101)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=101)
    param_grid = {"C": np.logspace(-4, +2, num=50)}

    # grid_search = GridSearchCV(lr,
    #                            param_grid=param_grid,
    #                            cv=cv,
    #                            scoring='f1')
    #
    # print 'Fit the grid search'
    # grid_search.fit(X, y)
    #
    # cv_results = grid_search.cv_results_
    # print cv_results
    #
    # best_estimator = grid_search.best_estimator_
    # print best_estimator
    #
    # best_score = grid_search.best_score_
    # print best_score
    #
    # best_params = grid_search.best_params_
    # print best_params
    #
    # scorer = grid_search.scorer_
    # print scorer
    #
    # f = open(DATA + "dict_f1.json", "w")
    # f.write(str(cv_results))
    # f.close()
    #
    #
    # grid_search = GridSearchCV(lr,
    #                            param_grid=param_grid,
    #                            cv=cv,
    #                            scoring='roc_auc')
    #
    # print 'Fit the grid search'
    # grid_search.fit(X, y)
    #
    # cv_results = grid_search.cv_results_
    # print cv_results
    #
    # best_estimator = grid_search.best_estimator_
    # print best_estimator
    #
    # best_score = grid_search.best_score_
    # print best_score
    #
    # best_params = grid_search.best_params_
    # print best_params
    #
    # scorer = grid_search.scorer_
    # print scorer
    #
    # f = open(DATA + "dict_roc_auc.json", "w")
    # f.write(str(cv_results))
    # f.close()

    grid_search = GridSearchCV(lr,
                               param_grid=param_grid,
                               cv=cv,
                               scoring='recall')

    print 'Fit the grid search'
    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_
    print cv_results

    best_estimator = grid_search.best_estimator_
    print best_estimator

    best_score = grid_search.best_score_
    print best_score

    best_params = grid_search.best_params_
    print best_params

    scorer = grid_search.scorer_
    print scorer

    f = open(DATA + "dict_recall.json", "w")
    f.write(str(cv_results))
    f.close()


def outputGridSearchLogisticRegression(path=DATA_LOCAL + 'accepted_refused_ds.csv'):

    data_file = open(DATA_LOCAL + 'dict_recall.json', 'r')
    # print data_file.read()
    data = json.load(data_file)
    # pprint(data)

    params = data['param_C']['data']
    acc = data['mean_test_score']
    best_param = params[acc.index(max(acc))]
    print 'Best param: ' + str(best_param)
    # log loss: 0.0281176869797
    # f1: 0.0281176869797
    # roc_auc: 0.0001
    # recall: 0.0001
    plt.semilogx(data['param_C']['data'], data['mean_test_score'])
    # plt.show()
    plt.close()
    print 'Data transformation'
    # my_data = dataTransformation(path)
    my_data = pd.read_csv(DATA_LOCAL + 'accepted_refused_ds_trans.csv', header=0)
    print my_data.head()
    y = my_data['loan']
    y = y.map(lambda x: str(x).replace('1', 'acc').replace('0', 'ref'))
    X = my_data.drop('loan', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#    #print sum(y_test)
    #print len(y_test) - sum(y_test)
    print 'Built and fit the model'
    lr = LogisticRegression(class_weight='balanced', random_state=101, C=best_param)
    lr.fit(X=X_train, y=y_train)
    y_pred = lr.predict_proba(X=X_test)
    coef = lr.coef_
    print lr.get_params()
    name_coef = my_data.columns.values[:-1]
    coef_val = pd.DataFrame(data={'coef': name_coef, 'val': coef[0]})
    coef_val.to_csv(DATA + 'LogisticRegressionCoef_c' + str(best_param) + '.csv', index=False)


    print 'Built ROC curve'
    # voglio avere valori = 0 per ACCETTATO
    #                     = 1 per REFUSED
    # cos' nella matroce di cinfusione ho TP FN
    #                                     FP TN
    y_acc = [p[1] for p in y_pred]
    y_test = y_test.map(lambda x: int(x.replace('acc', '0').replace('ref', '1')))
    fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_acc)
    # l = np.arange(len(tpr))
    # roc = pd.DataFrame(
    #     {'fpr': pd.Series(fpr, index=l),
    #      'tpr': pd.Series(tpr, index=l),
    #      '1-fpr': pd.Series(1 - fpr, index=l),
    #      'tf': pd.Series(tpr - (1 - fpr), index=l),
    #      'thresholds': pd.Series(threshold, index=l)}
    # )
    # print roc.ix[(roc.tf - 0.0).abs().argsort()[:1]]
    # # Plot tpr vs 1-fpr
    # fig, ax = plt.subplots()
    # plt.plot(roc['tpr'])
    # plt.plot(roc['1-fpr'], color='red')
    # plt.xlabel('1-False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # ax.set_xticklabels([])
    # plt.show()
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.8f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    # thre = float(roc.ix[(roc.tf - 0.0).abs().argsort()[:1]]['thresholds'])
    thre = 0.4

    source = ColumnDataSource(data=dict(tpr=tpr,
                                        fpr=fpr,
                                        thre=threshold
                                        ))
    TOOLS = "pan,wheel_zoom,reset,hover,save"
    p = Figure(tools=TOOLS)
    p.line('fpr', 'tpr', source=source, line_width=4)
    p.line([0, 1], [0, 1], line_dash='dashed', line_alpha=0.6)
    p.yaxis.axis_label = 'True Positive Rate'
    p.xaxis.axis_label = 'False Positive Rate'

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("FPR", "@fpr{1.11}"),
        ("TPR", "@tpr{1.11}"),
        ("THRESHOLD", "@thre{1.11}")
    ]

    show(p)

    y_pred_05 = [1 if a > 0.5 else 0 for a in y_acc]
    y_pred_thre = [1 if a > thre else 0 for a in y_acc]

    print 'Confusion matrix threshold = ' + str(thre)
    print confusion_matrix(y_true=y_test, y_pred=y_pred_thre)

    print 'Confusion matrix threshold = 0.5'
    print confusion_matrix(y_true=y_test, y_pred=y_pred_05)


    print 'Score f1 (thres = ' + str(thre) + ') = ' + str(f1_score(y_true=y_test, y_pred=y_pred_thre))

    print 'Score f1 (thres = 0.5) = ' + str(f1_score(y_true=y_test, y_pred=y_pred_05))

    print 'Score recall (thres = ' + str(thre) + ') = ' + str(recall_score(y_true=y_test, y_pred=y_pred_thre))

    print 'Score recall (thres = 0.5) = ' + str(recall_score(y_true=y_test, y_pred=y_pred_05))

    print 'Score accuracy (thres = ' + str(thre) + ') = ' + str(accuracy_score(y_true=y_test, y_pred=y_pred_thre))

    print 'Score accuracy (thres = 0.5) = ' + str(accuracy_score(y_true=y_test, y_pred=y_pred_05))

    print 'Score precision (thres = ' + str(thre) + ') = ' + str(precision_score(y_true=y_test, y_pred=y_pred_thre))

    print 'Score precision (thres = 0.5) = ' + str(precision_score(y_true=y_test, y_pred=y_pred_05))

    print 'Score AUC = ' + str(roc_auc)

    # import plot_roc as pr

    # pr.plot_roc(tpr=tpr,
    #             fpr=fpr,
    #             thresholds=threshold)


def test(path=DATA_LOCAL + "accepted_refused_ds.csv"):

    print 'Open data'
    data = pd.read_csv(path, header=0)
    data['month'] = [c[0:3] for c in data.date]
    print data.columns.values

    print 'Data transformation'
    feat = dataTransform(columns=['month', 'emp_len', 'state'],
                         applyTransformation=DictVectorizer(sparse=False),
                         columnsAppend=['amnt', 'dti', 'loan'])

    print 'Fit the data transformation'

    data_transformed = feat.fit_transform(data)

    # data_transformed.to_csv(DATA + 'accepted_refused_ds_small_trans.csv', index=False)

    return data_transformed


