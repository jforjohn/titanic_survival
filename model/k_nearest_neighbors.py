from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np
from MyFeatureSelection import MyFeatureSelection
import math



def mydist(x, y, w):
    return math.sqrt(np.sum(np.dot(w, (y - x) ** 2)))



def k_nearest_neighbors(data, labels, train_fidx, validation_fidx):

    w = MyFeatureSelection.compute_weights_relief(data,labels)

    knn = KNeighborsClassifier(
        '''n_neighbors=13'''
    )
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        knn.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = knn.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("K Nearest Neighbors", mean(folds_accuracy), "accuracy at validation stage.")

    return knn.fit(data, labels)

def weighted_k_nearest_neighbors(data, labels, train_fidx, validation_fidx):

    #w = MyFeatureSelection.compute_weights_relief(data, labels)
    w = MyFeatureSelection.compute_weights_info_gain(data, labels)

    knn = KNeighborsClassifier(
        #n_neighbors=9,
        metric=mydist, metric_params={"w": w}
    )
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        knn.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = knn.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("Weighted K Nearest Neighbors", mean(folds_accuracy), "accuracy at validation stage.")

    return knn.fit(data, labels)


def getBest(data, labels, train_fidx, validation_fidx):
    best_accuracy = 0
    best_params = None
    for n_neigh in range(1, 29, 1):
        knn = KNeighborsClassifier(
            n_neighbors=n_neigh
        )
        fold_accuracy_mean = 0
        for i in range(0, 30):

            folds_accuracy = list()
            for idx, trf in enumerate(train_fidx):

                knn.fit(data.loc[trf], labels.loc[trf])
                prediction_labels = knn.predict(data.loc[validation_fidx[idx]])

                folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
            fold_accuracy_mean = max(fold_accuracy_mean, mean(folds_accuracy))

        if fold_accuracy_mean > best_accuracy:
            best_accuracy = fold_accuracy_mean
            best_params = {'n_neighbors': n_neigh}
    print(best_params)
    print(best_accuracy)

def getBest_weighted(data, labels, train_fidx, validation_fidx):
    best_accuracy = 0
    best_params = None
    w = MyFeatureSelection.compute_weights_relief(data, labels)
    for n_neigh in range(1, 17, 2):
        print(n_neigh)
        knn = KNeighborsClassifier(
            n_neighbors=n_neigh, metric=mydist, metric_params={"w": w}
        )
        fold_accuracy_mean = 0
        for i in range(0, 1):
            folds_accuracy = list()
            for idx, trf in enumerate(train_fidx):

                knn.fit(data.loc[trf], labels.loc[trf])
                prediction_labels = knn.predict(data.loc[validation_fidx[idx]])

                folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
            fold_accuracy_mean = max(fold_accuracy_mean, mean(folds_accuracy))

        if fold_accuracy_mean > best_accuracy:
            best_accuracy = fold_accuracy_mean
            best_params = {'n_neighbors': n_neigh}
    print(best_params)
    print(best_accuracy)