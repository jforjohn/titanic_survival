from xgboost import XGBClassifier as xgb
from sklearn.metrics import accuracy_score
from statistics import mean

def xgBoost(data, labels, train_fidx, validation_fidx):
    model = xgb(colsample_bytree=0.5, learning_rate=0.1, max_depth=6, n_estimators=150, objective='binary:logistic', subsample=0.6)
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        model.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = model.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("XGBoost provided", mean(folds_accuracy), "accuracy at validation stage.")

    return model.fit(data, labels)

