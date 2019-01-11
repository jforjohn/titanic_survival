from sklearn.metrics import accuracy_score
from statistics import mean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA(data, labels, train_fidx, validation_fidx):

    lda = LinearDiscriminantAnalysis()
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        lda.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = lda.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("LDA", mean(folds_accuracy), "accuracy at validation stage.")

    return lda.fit(data, labels)

