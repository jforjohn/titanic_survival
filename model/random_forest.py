from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statistics import mean


def random_forest(data, labels, train_fidx, validation_fidx):
    rfc = RandomForestClassifier()
    #n_estimators=10
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        rfc.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = rfc.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("Random Forest provided", mean(folds_accuracy), "accuracy at validation stage.")

    return rfc.fit(data, labels)


def getBest(data, labels, train_fidx, validation_fidx):
    best_accuracy = 0
    best_params = None
    for n_est in range(10, 100, 5):
        rfc = RandomForestClassifier(
            n_estimators=n_est
        )
        fold_accuracy_mean = 0
        for i in range(0, 30):

            folds_accuracy = list()
            for idx, trf in enumerate(train_fidx):

                rfc.fit(data.loc[trf], labels.loc[trf])
                prediction_labels = rfc.predict(data.loc[validation_fidx[idx]])

                folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
            fold_accuracy_mean = max(fold_accuracy_mean, mean(folds_accuracy))

        if fold_accuracy_mean > best_accuracy:
            best_accuracy = fold_accuracy_mean
            best_params = {'n_estimators': n_est}
    print(best_params)
    print(best_accuracy)