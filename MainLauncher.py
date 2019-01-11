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
            df_train[col] = np.zeros([df_train.shape[0], 1])

        missing_cols = set(df_train.columns) - set(df_test.columns)
        for col in missing_cols:
            print(df_train.shape, df_test.shape)
            df_test[col] = np.zeros([df_test.shape[0], 1])

    labels_test = testPreprocess.labels_
    '''
    print()
    print('Train set sample')
    print(df_train.head())
    print()
    print('Test set sample')
    print(df_test.head())
    '''
    ##
    start = time()

    #models_perform(df_train, labels, df_test, labels_test)
    '''
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    kfold = KFold(n_splits=10)

    clf = svc = SVC(
        C=3,
        kernel='rbf',
        gamma='scale'
    )

    pred_model1 = clf.fit(df_train,labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model1)
    print(acc)
    res_model1 = cross_val_predict(clf, df_train, labels, cv=kfold)

    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,
        momentum=0.9,
        hidden_layer_sizes=(10, 50, 30, 40, 20),
        max_iter=750,
        random_state=42
    )
    pred_model2 = clf.fit(df_train, labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model2)
    print(acc)
    res_model2 = cross_val_predict(clf, df_train, labels, cv=kfold)

    from sklearn.ensemble import AdaBoostClassifier as ab
    clf = ab(learning_rate=0.5,
             n_estimators=300)
    pred_model3 = clf.fit(df_train, labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model3)
    print(acc)
    res_model3 = cross_val_predict(clf, df_train, labels, cv=kfold)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=10
    )
    pred_model4 = clf.fit(df_train, labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model4)
    print(acc)
    res_model4 = cross_val_predict(clf, df_train, labels, cv=kfold)

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    pred_model5 = clf.fit(df_train, labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model5)
    print(acc)
    res_model5 = cross_val_predict(clf, df_train, labels, cv=kfold)

    ens_model1 = np.concatenate(
        [res_model1.reshape(-1, 1),
         res_model2.reshape(-1, 1)], axis=1)

    ens_model2 = np.concatenate(
        [res_model3.reshape(-1, 1),
         res_model4.reshape(-1, 1),
         res_model5.reshape(-1, 1)], axis=1)

    ens_pred1 = np.concatenate(
        [pred_model1.reshape(-1, 1),
         pred_model2.reshape(-1, 1)], axis=1)

    ens_pred2 = np.concatenate(
        [pred_model3.reshape(-1, 1),
         pred_model4.reshape(-1, 1),
         pred_model5.reshape(-1, 1)], axis=1)

    from xgboost import XGBClassifier as xgb
    clf = xgb()
    preds_mid1 = clf.fit(ens_model1, labels).predict(ens_pred1)
    res_model_mid1 = cross_val_predict(clf, ens_model1, labels, cv=kfold)

    preds_mid2 = clf.fit(ens_model2, labels).predict(ens_pred2)
    res_model_mid2 = cross_val_predict(clf, ens_model2, labels, cv=kfold)

    ens_model_final = np.concatenate(
        [res_model_mid1.reshape(-1, 1),
         res_model_mid2.reshape(-1, 1)], axis=1)

    ens_pred_final = np.concatenate(
        [preds_mid1.reshape(-1, 1),
         preds_mid2.reshape(-1, 1)], axis=1)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=10
    )
    pred_model4 = clf.fit(df_train, labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model4)
    print(acc)
    res_model4 = cross_val_predict(clf, df_train, labels, cv=kfold)

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    pred_model5 = clf.fit(df_train, labels).predict(df_test)
    acc = accuracy_score(labels_test, pred_model5)
    print(acc)
    res_model5 = cross_val_predict(clf, df_train, labels, cv=kfold)

    ens_model1 = np.concatenate(
        [res_model1.reshape(-1, 1),
         res_model2.reshape(-1, 1)], axis=1)

    ens_model2 = np.concatenate(
        [res_model3.reshape(-1, 1),
         res_model4.reshape(-1, 1),
         res_model5.reshape(-1, 1)], axis=1)

    ens_pred1 = np.concatenate(
        [pred_model1.reshape(-1, 1),
         pred_model2.reshape(-1, 1)], axis=1)

    ens_pred2 = np.concatenate(
        [pred_model3.reshape(-1, 1),
         pred_model4.reshape(-1, 1),
         pred_model5.reshape(-1, 1)], axis=1)

    from xgboost import XGBClassifier as xgb
    clf = xgb()
    preds_mid1 = clf.fit(ens_model1, labels).predict(ens_pred1)
    res_model_mid1 = cross_val_predict(clf, ens_model1, labels, cv=kfold)

    preds_mid2 = clf.fit(ens_model2, labels).predict(ens_pred2)
    res_model_mid2 = cross_val_predict(clf, ens_model2, labels, cv=kfold)

    ens_model_final = np.concatenate(
        [res_model_mid1.reshape(-1, 1),
         res_model_mid2.reshape(-1, 1)], axis=1)

    ens_pred_final = np.concatenate(
        [preds_mid1.reshape(-1, 1),
         preds_mid2.reshape(-1, 1)], axis=1)

    #warnings.filterwarnings("ignore")
    '''
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

    from sklearn.linear_model import LogisticRegression
    #clf = LogisticRegression()
    preds = clf.fit(ens_model_final, labels).predict(ens_pred_final)
    acc = accuracy_score(labels_test, preds)
    print(acc)
