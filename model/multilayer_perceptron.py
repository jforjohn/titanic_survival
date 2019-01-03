from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from statistics import mean


#    net = MLPClassifier(
#        activation='relu',  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’   default relu
#        solver='adam',  # ‘lbfgs’, ‘sgd’, ‘adam’    default adam
#        learning_rate_init=0.0001,
#        momentum=0.9,
#        hidden_layer_sizes=(10, 10, 30, 10, 10)
#    )


configs = [
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.00001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.001, 'momentum':0.6, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.5, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.001, 'momentum':0.4, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.000001, 'momentum':0.5, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.7, 'hidden_layer_sizes':(10, 10, 10, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.7, 'hidden_layer_sizes':(10, 10, 30, 10, 10, 10, 10, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.8, 'hidden_layer_sizes':(30, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.00001, 'momentum':0.7, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.6, 'hidden_layer_sizes':(10, 20, 30, 15, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.00001, 'momentum':0.8, 'hidden_layer_sizes':(10, 10, 30, 20, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.7, 'hidden_layer_sizes':(10, 10, 30, 50, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10, 90, 20), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.00001, 'momentum':0.7, 'hidden_layer_sizes':(10, 10, 30, 10, 10, 40, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10, 30, 10, 20), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10, 10, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.00001, 'momentum':0.9, 'hidden_layer_sizes':(50, 40, 30, 20, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.9, 'hidden_layer_sizes':(10, 50, 30, 40, 20), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.01, 'momentum':0.6, 'hidden_layer_sizes':(10, 10, 30, 6, 5), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.0001, 'momentum':0.8, 'hidden_layer_sizes':(10, 10, 30, 10, 10), 'max_iter': 750},
{'activation':'relu', 'solver':'adam', 'learning_rate_init':0.00001, 'momentum':0.9, 'hidden_layer_sizes':(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30, 10, 10), 'max_iter': 750}
]


def mlp(data, labels, train_fidx, validation_fidx):
    net = MLPClassifier(
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,
        momentum=0.9,
        hidden_layer_sizes=(10, 50, 30, 40, 20)
    )
    folds_accuracy = list()
    for idx, trf in enumerate(train_fidx):
        net.fit(data.loc[trf], labels.loc[trf])
        prediction_labels = net.predict(data.loc[validation_fidx[idx]])

        folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
    print("Multilayer Perceptron provided", mean(folds_accuracy), "accuracy at validation stage.")

    return net.fit(data, labels)


def getBest(data, labels, train_fidx, validation_fidx):
    best_accuracy = 0
    best_params = None
    for params in configs:
        net = MLPClassifier(**params)
        fold_accuracy_mean = 0
        for i in range(0, 10):

            folds_accuracy = list()
            for idx, trf in enumerate(train_fidx):

                net.fit(data.loc[trf], labels.loc[trf])
                prediction_labels = net.predict(data.loc[validation_fidx[idx]])

                folds_accuracy.append(accuracy_score(labels.loc[validation_fidx[idx]], prediction_labels))
                print("Done")
            fold_accuracy_mean = max(fold_accuracy_mean, mean(folds_accuracy))

        if fold_accuracy_mean > best_accuracy:
            best_accuracy = fold_accuracy_mean
            best_params = params
    print(best_params)
    print(best_accuracy)