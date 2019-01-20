from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from statistics import mean


def svm(data, labels, train_fidx, validation_fidx):
    svc = SVC(C=1, gamma=0.1, kernel='rbf')
    ''' C=3,
    kernel='linear',
    gamma='auto '''
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        svc.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = svc.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("Support Vector Classifier provided", mean(folds_accuracy), "accuracy at validation stage.")

    return svc.fit(data, labels)

configs = [
    {'C': 3, 'kernel': 'linear', 'gamma': 'auto'},
{'C': 4, 'kernel': 'linear', 'gamma': 'auto'},
{'C': 2, 'kernel': 'linear', 'gamma': 'auto'},
{'C': 2.5, 'kernel': 'linear', 'gamma': 'auto'},
{'C': 2.4, 'kernel': 'linear', 'gamma': 'auto'},
{'C': 3, 'kernel': 'poly', 'gamma': 'auto'},
{'C': 2.3, 'kernel': 'poly', 'gamma': 'auto'},
{'C': 2, 'kernel': 'poly', 'gamma': 'auto'},
{'C': 3.2, 'kernel': 'poly', 'gamma': 'auto'},
{'C': 4, 'kernel': 'poly', 'gamma': 'auto'},
{'C': 2.7, 'kernel': 'poly', 'gamma': 'auto'},
{'C': 2, 'kernel': 'rbf', 'gamma': 'auto'},
{'C': 3, 'kernel': 'rbf', 'gamma': 'auto'},
{'C': 4, 'kernel': 'rbf', 'gamma': 'auto'},
{'C': 2.4, 'kernel': 'rbf', 'gamma': 'auto'},
{'C': 2.7, 'kernel': 'rbf', 'gamma': 'auto'},
{'C': 3.2, 'kernel': 'rbf', 'gamma': 'auto'},
{'C': 2, 'kernel': 'sigmoid', 'gamma': 'auto'},
{'C': 2.3, 'kernel': 'sigmoid', 'gamma': 'auto'},
{'C': 2.7, 'kernel': 'sigmoid', 'gamma': 'auto'},
{'C': 3, 'kernel': 'sigmoid', 'gamma': 'auto'},
{'C': 3.3, 'kernel': 'sigmoid', 'gamma': 'auto'},
{'C': 4, 'kernel': 'sigmoid', 'gamma': 'auto'},
]


def getBest(data, labels, train_fidx, validation_fidx):
    best_accuracy = 0
    best_params = None
    for params in configs:
        svc = SVC(**params)
        fold_accuracy_mean = 0
        for i in range(0, 30):

            folds_accuracy = list()
            for idx, trf in enumerate(train_fidx):

                svc.fit(data.loc[trf], labels.loc[trf])
                prediction_labels = svc.predict(data.loc[validation_fidx[idx]])

                folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
            fold_accuracy_mean = max(fold_accuracy_mean, mean(folds_accuracy))

        if fold_accuracy_mean > best_accuracy:
            best_accuracy = fold_accuracy_mean
            best_params = params
    print(best_params)
    print(best_accuracy)