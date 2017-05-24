import pandas as pd
from settings import DATA, DATA_LOCAL
from sklearn.feature_extraction import DictVectorizer
from sklearn import base
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt
import timeit
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

class dataTransform(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, columns, applyTransformation, columnsAppend):
        self.columns = columns
        self.applyTransformation = applyTransformation
        self.columnsAppend = columnsAppend

    def fit(self, X):
        self.applyTransformation.fit(X[self.columns].to_dict(orient='records'))
        # print X[self.columns].to_dict(orient='records')
        self.feature_names = self.applyTransformation.get_feature_names()
        print self.feature_names
        return self

    def transform(self, X):
        matrix = self.applyTransformation.transform(X[self.columns].to_dict(orient='records'))
        for column in self.columnsAppend:
            matrix = np.concatenate((matrix, X[column].values.reshape(1, -1).T), axis=1)
        columns_name = self.feature_names + self.columnsAppend
        matrix = pd.DataFrame(matrix, columns=columns_name).round(3)
        matrix[matrix.isnull()] = np.nan
        # matrix = matrix.apply(lambda x: x.str.strip()).replace('', np.nan)
        # to_int = self.feature_names + ['loan']
        # matrix[to_int] = matrix[to_int].astype(int)
        return matrix


class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
        # Fit the stored estimator.
        # Question: what should be returned?

    def transform(self, X):
        return np.array(self.estimator.predict(X)).reshape(-1, 1)


class dataFillNA(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, applyTransformation):
        self.applyTransformation = applyTransformation

    def fit(self, X):
        # self.applyTransformation.fit(X)
        return self

    def transform(self, X):
        matrix = X.dropna(how='all', axis=1)
        for x in matrix.columns:
            print x
            matrix[x] = matrix[x].astype(float)
        # matrix = matrix.astype(float)
        self.columns = matrix.columns
        matrix = self.applyTransformation.fit_transform(matrix)
        print self.columns
        print len(self.columns)
        # print len(matrix)
        matrix = pd.DataFrame(matrix)
        matrix.columns = self.columns
        return matrix


def dataTransformation(categorical_var, continous_var, path=DATA_LOCAL + "accepted.csv"):
    print 'Open data'
    data = pd.read_csv(path, header=0, low_memory=False)

    # data_train, data_test = train_test_split(data, test_size=0.001, random_state=101)


    # data = data_test

    data = data.dropna(subset=['int_rate'], axis=0)

    data['issue_d'] = map(lambda x: str(x)[0:3], data['issue_d'])
    # data['revol_util'] = map(lambda x: str(x)[:-1], data['revol_util'])
    data['int_rate'] = map(lambda x: float(str(x)[:-1]), data['int_rate'])
    print data.columns.values

    print 'Data transformation'

    feat = dataTransform(columns=categorical_var,
                         applyTransformation=DictVectorizer(sparse=False),
                         columnsAppend=continous_var)

    # fillNA = Imputer(missing_values=np.nan, strategy='mean', axis=1)
    fillNA = dataFillNA(applyTransformation=Imputer(missing_values=np.nan, strategy='mean'))
    # imputed_DF = pd.DataFrame(fill_NaN.fit_transform(DF))
    # imputed_DF.columns = DF.columns
    # imputed_DF.index = DF.index

    pipeline = Pipeline([('dataTransf', feat),
                         ('fillNA', fillNA)
                         ])

    print 'Fit the data transformation'

    # data_transformed = feat.fit_transform(data)

    data_transformed = pipeline.fit_transform(data)

    print data_transformed

    data_transformed.to_csv(DATA_LOCAL + 'accepted_trans.csv', index=False)


def data_cleaning():
    # data_small = data.ix[1:1000, ]
    # data_small.to_csv(DATA_LOCAL + 'accepted_small.csv', index=False)

    # data = pd.read_csv(DATA_LOCAL + 'accepted.csv', low_memory=False)
    # data = pd.read_csv(DATA_LOCAL + 'accepted_small.csv')

    CATEGORICAL_VAR = ['emp_length', 'home_ownership', 'verification_status', 'issue_d',
                       'addr_state', 'initial_list_status', 'application_type', 'verification_status_joint', 'purpose']

    CONTINUOUS_VAR = ['loan_amnt', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                      'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
                      'total_acc', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq', 'tot_coll_amt',
                      'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'total_bal_il',
                      'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi',
                      'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
                      'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
                      'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
                      'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
                      'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
                      'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
                      'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
                      'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort',
                      'total_bc_limit', 'total_il_high_credit_limit', 'il_util', 'int_rate']

    NLP = ['emp_title', 'desc', 'title']

    OTHERS_TRANS = ['zip_code']

    dep_var = 'int_rate'

    dataTransformation(categorical_var=CATEGORICAL_VAR,
                       continous_var=CONTINUOUS_VAR)


def featuresReduction(path=DATA_LOCAL + 'accepted_trans.csv'):
    print 'Open dataset'
    data = pd.read_csv(path)

    y = data['int_rate']
    X = data.drop('int_rate', axis=1)
    X_columns_name = X.columns.values
    # X = normalize(X, axis=0)
    # X = norm.fit_transform(X)
    X = X.apply(lambda x: (x - x.mean()) / (x.max()-x.min() + .1))
    print X.shape
    med = X.mean(axis=0)
    stdv = X.std(axis=0)
    print med
    print stdv



    print 'Fit Transform the model'

    pca = PCA(random_state=101, n_components=40)
    X_tr = pd.DataFrame(pca.fit_transform(X))
    X_tr = pd.concat([X_tr, y], axis=1)

    X_tr.to_csv(DATA_LOCAL + 'accepted_trans_red.csv', index=False)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    # df = pd.DataFrame({'explained_variance': explained_variance, 'explaned_variance_ratio': explained_variance_ratio})
    # df.to_csv(DATA_LOCAL + 'pca_explained_variance.csv', index=False)

    print explained_variance_ratio

    components = pd.DataFrame(pca.components_)
    components.columns = X_columns_name
    components.to_csv(DATA_LOCAL + 'pca_components.csv', index=False)

    plt.plot(range(len(explained_variance_ratio)), np.cumsum(explained_variance_ratio))
    plt.show()


def modelingSVR(path=DATA_LOCAL + 'accepted_trans_red.csv'):
    print 'Open data'
    data = pd.read_csv(path)

    y = data['int_rate']
    X = data.drop('int_rate', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # print 'Grid search'
    #
    # nysro = Nystroem(kernel='rbf')
    # model_lin_svr = svm.LinearSVR()
    # pipe = Pipeline([
    #     ('transformation', nysro),
    #     ('model', model_lin_svr)
    # ])

    # "transformation__kernel": ['rbf', 'poly', 'sigmoid', 'laplacian'],
    # "transformation__gamma": [0.01, 0.1, 1, 10],
    # "model__C": [0.01, 0.1, 1, 10],
    # "model__epsilon": [0.1, 0, 0.5, 10]
    # param_grid = {
    # "transformation__kernel": ['rbf'],
    # "transformation__gamma": [0.1, 1, 10],
    # "model__C": [0.01, 0.1, 1, 10],
    #    "model__epsilon": [0.1, 0, 0.5, 10]
    # }

    # cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=101)
    # grid_search = GridSearchCV(pipe,
    #                            param_grid=param_grid,
    #                            cv=cv,
    #                            scoring='neg_mean_squared_error')

    # grid_search.fit(X, y)

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
    # f = open(DATA_LOCAL + "grid_search_neg_mean_squared_error_rbf.json", "w")
    # f.write(str(cv_results))
    # f.close()


    kernel = ['laplacian'] #['rbf', 'laplacian', 'sigmoid', 'poly']#'sigmoid', 'laplacian'] # 'rbf', 'poly',
    gamma = [0.003, 0.004, 0.005, 0.006, 0.007]
    C = [7, 8, 9, 10, 11, 12, 13]

    mse = []
    r2 = []
    ker = []
    gam = []
    si = []

    for i in kernel:
        for j in gamma:
            for k in C:
                print 'Fit model'

                time1 = timeit.default_timer()
                nysro = Nystroem(kernel=i, gamma=j)
                model_lin_svr = svm.LinearSVR(C=k)
                pipe = Pipeline([
                    ('transformation', nysro),
                    ('model', model_lin_svr)
                ])
                pipe.fit(X_train, y_train)
                time2 = timeit.default_timer()

                print 'Fit time = ' + str(time2 - time1)

                print 'Predict model'
                time3 = timeit.default_timer()
                y_pred = pipe.predict(X_test)
                # X_test_feat = rbf_feature.transform(X_test)
                # y_pred = model_1.predict(X_test_feat)
                time4 = timeit.default_timer()

                print 'Predict time = ' + str(time4 - time3)

                mse_1 = mean_squared_error(y_test, y_pred)
                r2_1 = r2_score(y_test, y_pred)
                mse.append(mse_1)
                r2.append(r2_1)
                ker.append(i)
                gam.append(j)
                si.append(k)

                results = pd.DataFrame({'kernel': ker, 'gamma': gam, 'C': si, 'mse': mse, 'r2': r2})
                results.to_csv(DATA_LOCAL + 'gridSearch.csv', index=False)

                print str(i) + ' - ' + str(j) + ' - ' + str(k)

                #
                # err = y_pred - y_test
                # plt.plot(range(10000), err[0:10000])
                # plt.show()

                # value without kernel tarnsforations
                # 840.437519594
                # -42.7482555131


def modelingRandomForest(path=DATA_LOCAL + 'accepted_trans_red.csv'):

    print 'Open data'
    data = pd.read_csv(path)

    y = data['int_rate']
    X = data.drop('int_rate', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    n_estimators = [50, 100, 150, 200, 500] #['rbf', 'laplacian', 'sigmoid', 'poly']#'sigmoid', 'laplacian'] # 'rbf', 'poly',
    max_features = ['auto']
    min_samples_split = [60, 80, 100]

    mse = []
    r2 = []
    n_est = []
    max_feat = []
    min_samples_spl = []

    for i in n_estimators:
        for j in max_features:
            for k in min_samples_split:
                print 'Fit model'

                model_2 = RandomForestRegressor(random_state=101, n_estimators=i, max_features=j, min_samples_split=k)

                model_2.fit(X_train, y_train)
                y_pred = model_2.predict(X_test)

                mse_1 = mean_squared_error(y_test, y_pred)
                r2_1 = r2_score(y_test, y_pred)
                mse.append(mse_1)
                r2.append(r2_1)
                n_est.append(i)
                max_feat.append(j)
                min_samples_spl.append(k)


                results = pd.DataFrame({'n_estimators': n_est, 'max_features': max_feat,
                                        'min_samples_split': min_samples_spl, 'mse': mse, 'r2': r2})

                results.to_csv(DATA_LOCAL + 'gridSearchRandomForest.csv', index=False)

                print str(i) + ' - ' + str(j) + ' - ' + str(k)


def modelingKNN(path=DATA_LOCAL + 'accepted_trans_red.csv'):

    print 'Open data'
    data = pd.read_csv(path)

    y = data['int_rate']
    X = data.drop('int_rate', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    print y_test.var()

    n_neig = [200, 250] #['rbf', 'laplacian', 'sigmoid', 'poly']#'sigmoid', 'laplacian'] # 'rbf', 'poly',


    mse = []
    r2 = []
    n_neighbor = []
    # max_feat = []
    # min_samples_spl = []

    for i in n_neig:
        print str(i)
        print 'Fit model'
        model_2 = KNeighborsRegressor(n_neighbors=i)
        model_2.fit(X_train, y_train)
        y_pred = model_2.predict(X_test)
        mse_1 = mean_squared_error(y_test, y_pred)
        r2_1 = r2_score(y_test, y_pred)
        mse.append(mse_1)
        r2.append(r2_1)
        n_neighbor.append(i)

        results = pd.DataFrame({'n_neighbor': n_neighbor, 'mse': mse, 'r2': r2})
        results.to_csv(DATA_LOCAL + 'gridSearchKNN.csv', index=False)



def ensambleModel(path=DATA_LOCAL + 'accepted_trans_red.csv'):

    print 'Open data'
    data = pd.read_csv(path)

    y = data['int_rate']
    X = data.drop('int_rate', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    print np.var(y_test)
    # 19.2107664577

    nysro = Nystroem(kernel='laplacian', gamma=0.005)
    lin_svr = EstimatorTransformer(svm.LinearSVR(C=13))

    pipe_svr = Pipeline([
        ('nysro', nysro),
        ('lin_svr', lin_svr)
    ])

    rand_for = EstimatorTransformer(RandomForestRegressor(random_state=101, n_estimators=150, max_features='auto', min_samples_split=80))

    knn = EstimatorTransformer(KNeighborsRegressor(n_neighbors=250))

    ensamble = FeatureUnion([
        ('svr', pipe_svr),
        ('random_forest', rand_for),
        ('knn', knn)
    ])

    final_model = Pipeline([
        ('union', ensamble),
        ('kern', Nystroem(kernel='poly')),
        ('linear_reg', svm.LinearSVR())
    ])

    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    mse_1 = mean_squared_error(y_test, y_pred)
    r2_1 = r2_score(y_test, y_pred)

    print mse_1
    print r2_1

    # 14.2689852054
    # 0.257240192015



    # kernel = ['laplacian', 'sigmoid', 'poly']#'rbf',  'sigmoid', 'laplacian'] # 'rbf', 'poly',
    # gamma = [0.0001, 0.001, 0.01]
    # C = [0.0001, 0.001, 0.01]
    #
    # mse = []
    # r2 = []
    # ker = []
    # gam = []
    # si = []
    #
    # for i in kernel:
    #     for j in gamma:
    #         for k in C:
    #
    #             print 'Fit Data'
    #
    #             final_model = Pipeline([
    #                 ('union', ensamble),
    #                 ('kern', Nystroem(kernel=i, gamma=j)),
    #                 ('linear_reg', svm.LinearSVR(C=k))
    #             ])
    #
    #             final_model.fit(X_train, y_train)
    #
    #             y_pred = final_model.predict(X_test)
    #
    #
    #             mse_1 = mean_squared_error(y_test, y_pred)
    #             r2_1 = r2_score(y_test, y_pred)
    #             mse.append(mse_1)
    #             r2.append(r2_1)
    #             ker.append(i)
    #             gam.append(j)
    #             si.append(k)
    #
    #             print str(i) + ' - ' + str(j) + ' - ' + str(k)
    #             print 'mse = ' + str(mse_1)
    #             print 'r2 = ' + str(r2_1)
    #
    #             results = pd.DataFrame({'kernel': ker, 'gamma': gam, 'C': si, 'mse': mse, 'r2': r2})
    #             results.to_csv(DATA_LOCAL + 'gridSearchEnsamble.csv', index=False)


    # linear
    # mse = 16.791206623
    # r2 = 0.125948115607

    # nisqui
    # linear svr
    # mse = 17.3781059024
    # r2 = 0.0953975761132