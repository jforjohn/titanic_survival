from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def models_perform(train, test):

    # Divide dataset in folds
    kf = KFold(n_splits=5)
    train_idx, validation_idx = kf.split(train)

    # Each call to a model executes KFold validation
    # returns the model trained with the totality of the training data.
    models = list()
    # Multilayer perceptron

    # Support Vector Machine

    # Random Forest

    # XGD Boost

    # KNN

    # AdaBoost

    models_compare(models, test)


def models_compare(models, test):
    best_model = None
    best_accuracy = None
    for model in models:
        accuracy = accuracy_score(test.iloc[:, 0], model.predict(test.iloc[:, 1:]))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("Best model was", type(best_model).__name__, "with", best_accuracy, "accuracy")
