import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import sklearn_relief as relief
import pandas as pd



class MyFeatureSelection:

    # n is the number of features in the projected space
    @staticmethod
    def applyPCA(dataset_train, dataset_test, n):
        pca = PCA(n_components=n)
        pca.fit(dataset_train)
        transformed_train = pca.transform(dataset_train)
        transformed_test = pca.transform(dataset_test)

        transformed_train = pd.DataFrame(transformed_train)
        transformed_test = pd.DataFrame(transformed_test)

        return transformed_train, transformed_test

    # n is the number of features in the projected space
    @staticmethod
    def applyICA(dataset_train, dataset_test, n):
        ica = FastICA(n_components=n)
        ica.fit(dataset_train)
        transformed_train = ica.transform(dataset_train)
        transformed_test = ica.transform(dataset_test)

        transformed_train = pd.DataFrame(transformed_train)
        transformed_test = pd.DataFrame(transformed_test)

        return transformed_train, transformed_test


    @staticmethod
    def InfoGainSelection(dataset_train, dataset_test, labels_train, n):
        select_k = SelectKBest(mutual_info_classif, k=n)
        select_k.fit(dataset_train, labels_train)

        transformed_train = select_k.transform(dataset_train)
        transformed_test = select_k.transform(dataset_test)

        transformed_train = pd.DataFrame(transformed_train)
        transformed_test = pd.DataFrame(transformed_test)

        return transformed_train, transformed_test

    @staticmethod
    def AnovaSelection(dataset_train, dataset_test, labels_train, n):
        select_k = SelectKBest(f_classif, k=n)
        select_k.fit(dataset_train, labels_train)

        transformed_train = select_k.transform(dataset_train)
        transformed_test = select_k.transform(dataset_test)

        transformed_train = pd.DataFrame(transformed_train)
        transformed_test = pd.DataFrame(transformed_test)

        return transformed_train, transformed_test

        # Compute weights with information gain
        # train_set and train_labels must be pandas before converting to list for ib

    @staticmethod
    def compute_weights_info_gain(dataset, labels):
        r = mutual_info_classif(dataset, labels)

        return r

    @staticmethod
    def compute_weights_anova(dataset, labels):
        r = f_classif(dataset, labels)

        return r

    @staticmethod
    def compute_weights_relief(dataset, labels):

        # Relief
        r = relief.Relief(
            n_features=len(dataset.columns), n_jobs=1
        )

        r.fit_transform(
            np.array(dataset),
            np.array(labels)
        )

        return r.w_
