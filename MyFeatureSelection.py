import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import sklearn_relief as relief


class MyFeatureSelection:

    # n is the number of final features
    @staticmethod
    def applyPCA(dataset, n):
        pca = PCA(n_components=n)
        transformed = pca.fit_transform(dataset)
        return transformed

    @staticmethod
    def InfoGainSelection(dataset, labels, n):
        transformed = SelectKBest(mutual_info_classif, k=n).fit_transform(dataset, labels)
        return transformed

    @staticmethod
    def AnovaSelection(dataset, labels, n):
        transformed = SelectKBest(f_classif, k=n).fit_transform(dataset, labels)
        return transformed

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
