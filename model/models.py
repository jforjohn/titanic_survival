from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from model.multilayer_perceptron import mlp
from model.svm import svm
from model.random_forest import random_forest


def models_perform(data, data_labels):

    # Divide dataset in folds
    kf = KFold(n_splits=5)

    folds = [(train_idx, validation_idx) for train_idx, validation_idx in kf.split(data)]
    train_idx = [f[0] for f in folds]
    validation_idx = [f[1] for f in folds]

    # Each call to a model executes KFold validation
    # returns the model trained with the totality of the training data.
    models = list()
    # Multilayer perceptron
    mlp(data, data_labels, train_idx, validation_idx)
    # Support Vector Machine
    svm(data, data_labels, train_idx, validation_idx)
    # Random Forest
    random_forest(data, data_labels, train_idx, validation_idx)
    # XGD Boost

    # KNN

    # AdaBoost

    #models_compare(models,)


def models_compare(models, test):
    best_model = None
    best_accuracy = None
    for model in models:
        accuracy = accuracy_score(test.iloc[:, 0], model.predict(test.iloc[:, 1:]))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("Best model was", type(best_model).__name__, "with", best_accuracy, "accuracy")
