from enum import Enum
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import pandas as pd


class Model(Enum):
    DECISION_TREE = 1
    RANDOM_FOREST = 2
    ADABOOST = 3
    GRADIENT_BOOSTING = 4
    XGBOOST = 5
    SVM_LINEAR = 6
    SVM_POLY = 7
    SVM_SIGMOID = 8
    SVM_RBF = 9
    LOGISTIC_REGRESSION = 10
    NAIVE_BAYES_GAUSSIAN = 11
    NAIVE_BAYES_MULTINOMIAL = 12
    SGD = 13
    KNN = 14
    MLP = 15


def train_decision_tree(*, features, label, preprocessor, train_x, train_y, depth=2, leaf=2, split=2,
                        class_weights=None, seed=1, return_tree=False):
    dt = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=leaf,
        min_samples_split=split,
        random_state=seed,
        class_weight=class_weights
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('decision_tree', dt)
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    if return_tree:
        pipeline.fit(x_train, y_train.values.ravel())
        return pipeline.named_steps['decision_tree']

    return pipeline.fit(x_train, y_train.values.ravel())


def train_random_forest(*, features, label, preprocessor, train_x=None, train_y=None, estimators=10, depth=2,
                        leaf=2, split=2, class_weights=None, seed=1, return_pipe=False):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('rf', RandomForestClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                min_samples_split=split,
                n_estimators=estimators,
                # max_features='sqrt',
                random_state=seed,
                class_weight=class_weights,
                # bootstrap=False,
                # criterion="entropy",
                n_jobs=-1
            ))
        ]
    )

    if return_pipe:
        return pipeline

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_adaboost(*, features, label, preprocessor, train_x, train_y, seed=1, estimators=3, depth=3, rate=1):
    svc = DecisionTreeClassifier(max_depth=depth)

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('ada',
             AdaBoostClassifier(
                 n_estimators=estimators,
                 estimator=svc,
                 learning_rate=rate,
                 random_state=seed,
                 algorithm='SAMME'
             )
             )
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_gradient_boosting(*, features, label, preprocessor, train_x, train_y, learning_rate=0.1, max_depth=3,
                            n_estimators=10, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('gradient_boosting',
             GradientBoostingClassifier(
                 learning_rate=learning_rate,
                 max_depth=max_depth,
                 n_estimators=n_estimators,
                 random_state=seed
             ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_xgboost(*, features, label, preprocessor, train_x, train_y, colsample_bytree=0.6, max_depth=7,
                  gamma=0.2, min_child_weight=1, subsample=0.9, include_class_weights=False, seed=1):
    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    neg_pos_ratio = None

    if include_class_weights:
        neg_count = (train_y == 0).sum()
        pos_count = (train_y == 1).sum()

        neg_pos_ratio = neg_count / pos_count

    model = XGBClassifier(
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        gamma=gamma,
        min_child_weight=min_child_weight,
        subsample=subsample,
        objective='binary:logistic',
        scale_pos_weight=neg_pos_ratio,
        seed=seed,
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('xgboost', model)
        ]
    )

    return pipeline.fit(x_train, y_train.values.ravel())


def train_svm_linear(*, features, label, preprocessor, train_x, train_y, c=1, tol=1e-2, probability=True,
                     class_weights=None, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('svm', SVC(
                kernel="linear",
                C=c,
                tol=tol,
                class_weight=class_weights,
                probability=probability,
                random_state=seed
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_svm_poly(*, features, label, preprocessor, train_x, train_y, c=1, degree=1, tol=1e-2, probability=True,
                   class_weights=None, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('svm', SVC(
                kernel="poly",
                C=c,
                degree=degree,
                tol=tol,
                class_weight=class_weights,
                probability=probability,
                random_state=seed
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_svm_sigmoid(*, features, label, preprocessor, train_x, train_y, c=1, gamma='auto', coef0=1, tol=1e-2,
                      probability=True, class_weights=None, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('svm', SVC(
                kernel="sigmoid",
                C=c,
                tol=tol,
                gamma=gamma,
                coef0=coef0,
                probability=probability,
                class_weight=class_weights,
                random_state=seed
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_svm_rbf(*, features, label, preprocessor, train_x, train_y, c=1, gamma='auto', coef0=1, tol=1e-2,
                  probability=True, class_weights=None, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('svm', SVC(
                kernel="rbf",
                C=c,
                tol=tol,
                gamma=gamma,
                coef0=coef0,
                class_weight=class_weights,
                probability=probability,
                random_state=seed
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_logistic_regression(*, features, label, preprocessor, train_x, train_y, c=1, solver="lbfgs", penalty="l1",
                              max_iter=100, class_weights=None, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier',
             LogisticRegression(
                 C=c,
                 solver=solver,
                 penalty=penalty,
                 max_iter=max_iter,
                 class_weight=class_weights,
                 random_state=seed,
             )
             )
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_naive_bayes_gaussian(*, features, label, preprocessor, train_x, train_y, var_smoothing=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('nb_gaussian', GaussianNB(
                var_smoothing=var_smoothing
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_naive_bayes_multinomial(*, features, label, preprocessor, train_x, train_y):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('nb_multinomial', MultinomialNB())
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train)


def train_sgd(*, features, label, preprocessor, train_x, train_y, seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('sgd', SGDClassifier(loss='log_loss', alpha=0.01,
                                  max_iter=1000, random_state=seed))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_knn(*, features, label, preprocessor, train_x, train_y, k=1, leaf_size=20, p=2, weights='uniform',
               algorithm='auto', metric='minkowski'):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('knn', KNeighborsClassifier(
                leaf_size=leaf_size,
                n_neighbors=k,
                p=p,
                weights=weights,
                algorithm=algorithm,
                metric=metric
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())


def train_mlp(*, features, label, preprocessor, train_x, train_y, max_iter=5000, activation='relu', solver='sgd',
              learning_rate_init=0.01, alpha=0.001, learning_rate='constant', hidden_layers=(6, 3, 2), seed=1):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('mlp', MLPClassifier(
                max_iter=max_iter,
                activation=activation,
                solver=solver,
                learning_rate_init=learning_rate_init,
                alpha=alpha,
                learning_rate=learning_rate,
                hidden_layer_sizes=hidden_layers,
                random_state=seed
            ))
        ]
    )

    x_train = pd.DataFrame(train_x, columns=features)
    y_train = pd.DataFrame(train_y, columns=[label])

    return pipeline.fit(x_train, y_train.values.ravel())
