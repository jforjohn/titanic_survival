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

    '''
    Best Feature Selection technique for each model:
                                        Accuracy    Time
    LR_d_45/49RandomForestClassifier	0.823334	0.021874
    ALL_49_XGBClassifier				0.822198	0.140709
    ALL_49_SVC							0.821009	0.033901
    ALL_49_LinearDiscriminantAnalysis	0.814099	0.018390
    ALL_49_KNeighborsClassifier			0.807196	0.015391
    AN_d_35/49MLPClassifier				0.808332	3.984188
    PCA_d_16/49_ev_0.885_MyIBL			0.778334	2.536446
    '''

    # LR - Lasso Regression Feature Selection - chooses 45 dimensions
    alpha = 0.00001
    lr_train, lr_test = MyFeatureSelection.LassoRegressionSelection(df_train, df_test, labels, alpha)

    # PCA - 16 dimensions
    n_dim = 16
    pca_train, pca_test, ev = MyFeatureSelection.applyPCA(df_train, df_test, n_dim)

    # AN - Anova Selection
    an_train, an_test = MyFeatureSelection.AnovaSelection(df_train, df_test, labels, n_dim)

