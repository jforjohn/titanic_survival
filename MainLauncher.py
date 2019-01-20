##
#from MyPreprocessing import MyPreprocessing
import pandas as pd
import numpy as np
from config_loader import load
import argparse
import sys
import seaborn as sns
from MyDataUnderstanding import featureAnalysis
from MyPreprocessing import MyPreprocessing
from MyFeatureSelection import MyFeatureSelection


import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from model.models import models_perform
from model.MyIBL import MyIBL
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def getData(path, filenames_type):
    '''
    features_lst = [
        "Pclass", "Survived", "Name", "Sex", "Age",
        "Sibsp", "Parch", "Ticket", "Fare", "Cabin","Embarked"]
    '''
    if filenames_type == 'train':
        filename = 'train'
    elif filenames_type == 'test':
        filename = 'test'
    else:
        filename = 'titanicAll'
    df_features = pd.read_csv(path + filename + '.csv',
                           sep=',')

    if filenames_type not in ['train', 'test']:
        # drop unnecessary columns that don't exist in the official dataset
        df_features.drop(['Boat', 'Body', 'Home.dest'],
                          axis=1,
                         inplace=True)
    #labels = df_features['Survived']
    #df_features = df_features.drop(['Survived'], axis=1)
    return df_features
##
if __name__ == '__main__':
    ##
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="titanic.cfg",
        help="specify the location of the clustering config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    ##
    verbose = config.get('titanic', 'verbose')
    path = config.get('titanic', 'path') + '/'
    file_type = config.get('titanic', 'file_type')

    filename_type = 'train'
    if file_type == 'all':
        filename_type = 'other'

    print('Filename type:', filename_type)
    print()
    ## train
    trainData = getData(path, filename_type)
    # Preprocessing
    trainPreprocess = MyPreprocessing(process_type='all',
                                      filename_type=filename_type,
                                      remove_outliers=True)

    ## test
    filename_type = 'test'
    testData = getData(path, filename_type)
    # Preprocessing
    testPreprocess = MyPreprocessing(process_type='all',
                                     filename_type=filename_type,
                                     remove_outliers=False)

    ## Data Understanding
    if verbose == 'true':
        featureAnalysis(trainData)
        featureAnalysis(testData)


    trainPreprocess.fit(trainData)
    df_train = trainPreprocess.new_df
    # the labels "Survived"
    labels = trainPreprocess.labels_
    #print(labels.head())
    # the initial dataset without any preprocessing
    #print(trainPreprocess.df_initial.head())
    # the preprocessed data
    #print(trainPreprocess.new_df.head())

    testPreprocess.fit(testData)
    df_test = testPreprocess.new_df

    # fix missing columns because of NaNs and one hot encoding without dummy_na
    if df_train.shape[1] != df_test.shape[1]:
        missing_cols = set(df_test.columns) - set(df_train.columns)
        for col in missing_cols:
            #df_train[col] = np.zeros([df_train.shape[0], 1])
            df_test.drop([col], axis=1, inplace=True)

        missing_cols = set(df_train.columns) - set(df_test.columns)
        for col in missing_cols:
            #df_test[col] = np.zeros([df_test.shape[0], 1])
            df_train.drop([col], axis=1, inplace=True)

    labels_test = testPreprocess.labels_

    print(df_train.columns, df_test.columns)
    print(df_train.shape, df_test.shape)

    # Calculate validation accuracy
    #models_perform(df_train,labels,df_test,labels_test)

    '''
    Best Feature Selection technique for each model:
                                            Accuracy    Time
    LR_d_18/49XGBClassifier						0.844595	0.071340
    IG_d_45/49SVC								0.834463	0.029715
    ICA_d_20/49MLPClassifier					0.829988	1.281857
    LR_d_30/49LinearDiscriminantAnalysis		0.831092	0.011121
    PCA_d_19/49_ev_0.934_KNeighborsClassifier	0.826617	0.007995
    IG_d_25/49RandomForestClassifier			0.816460	0.023214
    AN_d_35/49MyIBL								0.757925	2.434744
    '''


    # IG - Information Gain for  Feature Selection - chooses 45 dimensions - for feeding SVC
    num_dim_IG_SVC = 45
    ig_train_SVC, ig_test_SVC = MyFeatureSelection.InfoGainSelection(df_train, df_test, labels, num_dim_IG_SVC)

    # IG - Information Gain for  Feature Selection - chooses 25 dimensions - for feeding Random Forest
    num_dim_IG_RF = 25
    IG_train_RF, ig_test_RF = MyFeatureSelection.InfoGainSelection(df_train, df_test, labels, num_dim_IG_RF)

    # LR - Lasso Regression for  Feature Selection - chooses 18 dimensions - for feeding XGBoost
    alpha_XGB = 0.003
    lr_train_XGB, lr_test_XGB = MyFeatureSelection.LassoRegressionSelection(df_train, df_test, labels, alpha_XGB)

    # LR - Lasso Regression for  Feature Selection - chooses 30 dimensions - for feeding LDA
    alpha_LDA = 0.0009
    lr_train_LDA, lr_test_LDA = MyFeatureSelection.LassoRegressionSelection(df_train, df_test, labels, alpha_LDA)

    # ICA - ICA for  Feature Selection - chooses 20 dimensions - for feeding MLP
    num_dim_ICA = 20
    ica_train, ica_test = MyFeatureSelection.applyICA(df_train, df_test, num_dim_ICA)


    # PCA - PCA for  Feature Selection - chooses 19 dimensions - for feeding KNN
    num_dim_PCA = 19
    pca_train, pca_test, ev = MyFeatureSelection.applyPCA(df_train, df_test, num_dim_PCA)


    # AN - AN for  Feature Selection - chooses 35 dimensions - for feeding IBL
    num_dim_AN = 35
    an_train, an_test = MyFeatureSelection.AnovaSelection(df_train, df_test, labels, num_dim_AN)

    '''
    # AN - Anova Selection for  Feature Selection - chooses 10 dimensions - for feeding IB2
    num_dim_AN = 10
    an_train, an_test = MyFeatureSelection.AnovaSelection(df_train, df_test, labels, num_dim_AN)
    ibl = MyIBL(n_neighbors=9, ibl_algo='ib2', voting='mp', distance='euclidean')
    ibl.fit(an_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": ibl.predict(an_test)}).to_csv(
        './submissions/ibl.csv', index=False)

    # ICA - 20 dimensions - for feeding Random Forest Classifier
    n_dim_ICA = 20
    ica_train, ica_test = MyFeatureSelection.applyICA(df_train, df_test, n_dim_ICA)
    rfc = RandomForestClassifier(criterion='gini', max_depth=90, max_features='log2', min_samples_leaf=5, min_samples_split=8, n_estimators=200)
    rfc.fit(ica_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": rfc.predict(ica_test)}).to_csv('./submissions/rfc.csv', index=False)

    # PCA - 17 dimensions - for feeding MLP
    n_dim_PCA = 17
    pca_train, pca_test, ev = MyFeatureSelection.applyPCA(df_train, df_test, n_dim_PCA)
    net = MLPClassifier(max_iter=1000, activation='tanh', hidden_layer_sizes=14, learning_rate='constant', learning_rate_init=0.1, solver='sgd')
    net.fit(pca_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": net.predict(pca_test)}).to_csv('./submissions/mlp.csv', index=False)


    # Other
    xgb = XGBClassifier(colsample_bytree=0.6, learning_rate=0.1, max_depth=4, n_estimators=350, objective='binary:logistic', subsample=0.5)
    xgb.fit(df_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": xgb.predict(df_test)}).to_csv('./submissions/xgb.csv', index=False)


    lda = LinearDiscriminantAnalysis()
    lda.fit(df_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": lda.predict(df_test)}).to_csv('./submissions/lda.csv', index=False)


    svc = SVC(C=1, gamma=0.1, kernel='rbf')
    svc.fit(df_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": svc.predict(df_test)}).to_csv('./submissions/svc.csv', index=False)


    knn = KNeighborsClassifier(algorithm='brute', metric='euclidean', n_neighbors=9, weights='uniform')
    knn.fit(df_train, labels)
    pd.DataFrame({"PassengerId": testData["PassengerId"], "Survived": knn.predict(df_test)}).to_csv('./submissions/knn.csv', index=False)
    '''
