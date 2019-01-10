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
        "Sibsp", "Parch", "Ticket", "Fare", "Cabin","Embarked",
        "Boat","Body","Home.dest"]
    '''
    if filenames_type == 'train':
        filename = 'train'
    elif filenames_type == 'test':
        filename = 'test'
    else:
        filename = 'titanicAll'
    df_features = pd.read_csv(path + filename + '.csv',
                           sep=',')

    if filename not in ['train', 'test']:
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
    trainPreprocess = MyPreprocessing(filename_type)

    ## test
    filename_type = 'test'
    testData = getData(path, filename_type)
    # Preprocessing
    testPreprocess = MyPreprocessing(filename_type)

    ## Data Understanding
    if verbose == 'true':
        featureAnalysis(trainData)
        featureAnalysis(testData)


    trainPreprocess.fit(trainData)
    df_train = trainPreprocess.new_df
    # the labels "Survived"
    labels = trainPreprocess.labels_
    print(labels.head())
    # the initial dataset without any preprocessing
    print(trainPreprocess.df_initial.head())
    # the preprocessed data
    print(trainPreprocess.new_df.head())

    testPreprocess.fit(testData)
    df_test = testPreprocess.new_df
    print()
    print('Train set sample')
    print(df_train.head())
    print()
    print('Test set sample')
    print(df_test.head())

    ##
    start = time()

    df_test = df_train.iloc[891:]
    df_train = df_train.iloc[0:890]
    labels_test = labels.iloc[891:]
    labels = labels[0:890]

    #warnings.filterwarnings("ignore")

    print('Original')
    print('##################################')
    models_perform(df_train, labels, df_test, labels_test)

    print('')
    print('PCA')
    print('##################################')
    for n_dim in range(8, len(df_train.columns)):
        print('')
        print(n_dim, ' dimensions:')
        pca_train, pca_test = MyFeatureSelection.applyPCA(df_train, df_test, n_dim)
        models_perform(pca_train, labels, pca_test, labels_test)

    print('')
    print('ICA')
    print('##################################')
    for n_dim in range(8, len(df_train.columns)):
        print('')
        print(n_dim, ' dimensions:')
        ica_train, ica_test = MyFeatureSelection.applyICA(df_train, df_test, n_dim)
        models_perform(ica_train, labels, ica_test, labels_test)

    print('')
    print('ICA')
    print('##################################')
    for n_dim in range(8, len(df_train.columns)):
        print('')
        print(n_dim, ' dimensions:')
        ica_train, ica_test = MyFeatureSelection.applyICA(df_train, df_test, n_dim)
        models_perform(ica_train, labels, ica_test, labels_test)

    print('')
    print('INFO GAIN SELECTION')
    print('##################################')
    for n_dim in range(8, len(df_train.columns)):
        print('')
        print(n_dim, ' dimensions:')
        ig_train, ig_test = MyFeatureSelection.InfoGainSelection(df_train, df_test, labels, n_dim)
        models_perform(ig_train, labels, ig_test, labels_test)

    print('')
    print('ANOVA SELECTION')
    print('##################################')
    for n_dim in range(8, len(df_train.columns)):
        print('')
        print(n_dim, ' dimensions:')
        an_train, an_test = MyFeatureSelection.AnovaSelection(df_train, df_test, labels, n_dim)
        models_perform(an_train, labels, an_test, labels_test)