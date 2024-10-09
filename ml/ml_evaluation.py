import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix


def predict(features, model, test):
    x_test = pd.DataFrame(test, columns=features)
    predictions = model.predict(x_test)
    return predictions


def evaluate_report(predictions, test):
    print('Classification Report: \n', classification_report(test, predictions))


def print_confusion_matrix(predictions, test):
    print('Confusion Matrix: \n', confusion_matrix(test, predictions))


def evaluate(label, predictions, test):
    y_test = pd.DataFrame(test, columns=[label])
    return accuracy_score(y_test, predictions)


def evaluate_full(label, predictions, test):
    y_test = pd.DataFrame(test, columns=[label])
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    return accuracy, f1


def calculate_accuracy(submission_text, ground_truth_csv):
    submission_lines = submission_text.split('\n')
    ground_truth_lines = [row[0] for row in ground_truth_csv]
    total_lines = len(submission_lines)
    correct_lines = sum(1 for sub, gt in zip(submission_lines, ground_truth_lines) if sub.strip() == gt.strip())
    accuracy = (correct_lines / total_lines) * 100
    return accuracy


def split_dataset(dataset, features, label, test_participants_all, n):
    test_participants = test_participants_all[n]

    dataset_train = dataset[~dataset['Participant'].isin(test_participants)]
    dataset_test = dataset[dataset['Participant'].isin(test_participants)]

    return dataset_train[features], dataset_test[features], dataset_train[label], dataset_test[label]
