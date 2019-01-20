from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from model.multilayer_perceptron import mlp
from model.svm import svm
from model.random_forest import random_forest
from model.k_nearest_neighbors import k_nearest_neighbors
from model.k_nearest_neighbors import weighted_k_nearest_neighbors
from model.LDA import LDA
from model.xgBoost import xgBoost
from MyFeatureSelection import MyFeatureSelection


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
    num_dim_ICA = 20
    ica_train, ica_test = MyFeatureSelection.applyICA(data, test, num_dim_ICA)
    models.append(mlp(ica_train, data_labels, train_idx, validation_idx))
    # Support Vector Machine
    num_dim_IG_SVC = 45
    ig_train_SVC, ig_test_SVC = MyFeatureSelection.InfoGainSelection(data, test, data_labels, num_dim_IG_SVC)
    models.append(svm(ig_train_SVC, data_labels, train_idx, validation_idx))
    # Random Forest
    num_dim_IG_RF = 25
    IG_train_RF, ig_test_RF = MyFeatureSelection.InfoGainSelection(data, test, data_labels, num_dim_IG_RF)
    models.append(random_forest(IG_train_RF, data_labels, train_idx, validation_idx))
    # XG Boost
    alpha_XGB = 0.003
    lr_train_XGB, lr_test_XGB = MyFeatureSelection.LassoRegressionSelection(data, test, data_labels, alpha_XGB)
    models.append(xgBoost(lr_train_XGB, data_labels, train_idx, validation_idx))
    # KNN
    num_dim_PCA = 19
    pca_train, pca_test, ev = MyFeatureSelection.applyPCA(data, test, num_dim_PCA)
    models.append(k_nearest_neighbors(pca_train, data_labels, train_idx, validation_idx))
    # Weighted KNN
    #models.append(weighted_k_nearest_neighbors(data, data_labels, train_idx, validation_idx))

    #LDA
    alpha_LDA = 0.0009
    lr_train_LDA, lr_test_LDA = MyFeatureSelection.LassoRegressionSelection(data, test, data_labels, alpha_LDA)
    models.append(LDA(lr_train_LDA, data_labels, train_idx, validation_idx))

    # AdaBoost



    #models_compare(models, test, test_labels)

def models_compare(models, test, test_labels):
    best_model = None
    best_accuracy = 0
    for model in models:
        accuracy = accuracy_score(test_labels, model.predict(test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("Best model was", type(best_model).__name__, "with", best_accuracy, "accuracy")
