from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from model.multilayer_perceptron import mlp
from model.svm import svm
from model.random_forest import random_forest
from model.k_nearest_neighbors import k_nearest_neighbors
from model.k_nearest_neighbors import weighted_k_nearest_neighbors
from model.LDA import LDA





def models_perform(data, data_labels, test, test_labels):

    # Divide dataset in folds
    kf = KFold(n_splits=5)

    folds = [(train_idx, validation_idx) for train_idx, validation_idx in kf.split(data)]
    train_idx = [f[0] for f in folds]
    validation_idx = [f[1] for f in folds]

    # Each call to a model executes KFold validation
    # returns the model trained with the totality of the training data.
    models = list()
    # Multilayer perceptron
    models.append(mlp(data, data_labels, train_idx, validation_idx))
    # Support Vector Machine
    models.append(svm(data, data_labels, train_idx, validation_idx))
    # Random Forest
    models.append(random_forest(data, data_labels, train_idx, validation_idx))
    # XGD Boost

    # KNN
    models.append(k_nearest_neighbors(data, data_labels, train_idx, validation_idx))
    # Weighted KNN
    models.append(weighted_k_nearest_neighbors(data, data_labels, train_idx, validation_idx))

    #LDA
    models.append(LDA(data, data_labels, train_idx, validation_idx))


    # AdaBoost

    models_compare(models, test, test_labels)


def models_compare(models, test, test_labels):
    best_model = None
    best_accuracy = 0
    for model in models:
        accuracy = accuracy_score(test_labels, model.predict(test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("Best model was", type(best_model).__name__, "with", best_accuracy, "accuracy")
