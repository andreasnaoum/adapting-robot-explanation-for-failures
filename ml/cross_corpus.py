import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

test_participants_all_codes = {
    1: ["C1-1"], 2: ["C1-2"], 3: ["C1-3"], 4: ["C1-4"], 5: ["C1-5"],
    6: ["C1-6"], 7: ["C1-7"], 8: ["C1-8"], 9: ["C1-9"], 10: ["C1-10"],
    11: ["C1-11"], 12: ["C2-1"], 13: ["C2-2"], 14: ["C2-3"], 15: ["C2-4"],
    16: ["C2-5"], 17: ["C2-6"], 18: ["C2-7"], 19: ["C2-8"], 20: ["C2-9"],
    21: ["C2-10"], 22: ["C2-11"], 23: ["C3-1"], 24: ["C3-2"], 25: ["C3-3"],
    26: ["C3-4"], 27: ["C3-5"], 28: ["C3-6"], 29: ["C3-7"], 30: ["C3-8"],
    31: ["C3-9"], 32: ["C3-10"], 33: ["C3-11"], 34: ["D1-1"], 35: ["D1-2"],
    36: ["D1-3"], 37: ["D1-4"], 38: ["D1-5"], 39: ["D1-6"], 40: ["D1-7"],
    41: ["D1-8"], 42: ["D1-9"], 43: ["D1-10"], 44: ["D1-11"], 45: ["D2-1"],
    46: ["D2-2"], 47: ["D2-3"], 48: ["D2-4"], 49: ["D2-5"], 50: ["D2-6"],
    51: ["D2-7"], 52: ["D2-8"], 53: ["D2-9"], 54: ["D2-10"], 55: ["D2-11"],
}


def show_results(final_accuracies, final_f1s):
    num_reps = len(final_accuracies)
    data = pd.DataFrame({
        'Participant': [f'P{i + 1}' for i in range(num_reps)] * 2,
        'Score': final_accuracies + final_f1s,
        'Metric': ['Accuracy'] * num_reps + ['F1 Score'] * num_reps
    })
    plt.figure(figsize=(25, 6))
    sns.barplot(x='Participant', y='Score', hue='Metric', data=data, palette="viridis")
    plt.xlabel('Test Participant')
    plt.ylabel('Scores')
    plt.title('Final Accuracies and F1 Scores by Participant')
    plt.show()


def split_dataset(dataset, features, label, n):
    test_participants = test_participants_all_codes[n]
    dataset_train = dataset[~dataset['Participant'].isin(test_participants)]
    dataset_test = dataset[dataset['Participant'].isin(test_participants)]
    return dataset_train[features], dataset_test[features], dataset_train[label], dataset_test[label]


def readData(file, numerical_features, features, label):
    dataset = pd.read_csv(file)

    dataset_confusion = dataset[dataset['Confusion?'] == True]
    dataset_not_confusion = dataset[dataset['Confusion?'] == False]
    count_all = len(dataset)
    count_confusion = len(dataset_confusion)
    count_not_confusion = len(dataset_not_confusion)
    print(f"Count of Confused Instances: {count_confusion}, Percentage: {count_confusion / count_all}")
    print(f"Count of Not Confused Instances: {count_not_confusion}, Percentage: {count_not_confusion / count_all}")

    dataset['Action'] = dataset["Action"].map(
        {
            'Pick': 0.3,
            'Carry': 0.6,
            'Place': 0.9,
        }
    )

    dataset['Confusion?'] = dataset["Confusion?"].map(
        {
            False: 0,
            True: 1
        }
    )

    dataset['Decrease Explanation Level'] = dataset["Decrease Explanation Level"].map(
        {
            False: 0,
            True: 1,
        }
    )

    dataset['Last Hand touching Face / Head'] = dataset['Last Hand touching Face / Head'].map({
        False: 0,
        True: 1,
    })

    dataset['Hand touching Face / Head Failure'] = dataset['Hand touching Face / Head Failure'].map({
        False: 0,
        True: 1,
    })

    dataset['Last Head Titling'] = dataset['Last Head Titling'].map({
        False: 0,
        True: 1,
    })

    dataset['Head Titling Failure'] = dataset['Head Titling Failure'].map({
        False: 0,
        True: 1,
    })

    return dataset[["Participant"] + features + [label]]


def preprocess(categorical_feature, boolean_feature, numerical_features, gesture_boolean):
    return ColumnTransformer(
        transformers=[
            ('action', 'passthrough', categorical_feature),
            ('decrease', 'passthrough', boolean_feature),
            ('reaction', 'passthrough', numerical_features),
            ('reaction1', 'passthrough', gesture_boolean),
        ])


seed = {1: 60, 2: 0, 3: 0, 4: 0, 5: 8, 6: 20, 7: 0, 8: 27, 9: 0, 10: 68, 11: 180, 12: 239, 13: 6, 14: 95, 15: 22, 16: 0, 17: 1, 18: 0, 19: 32, 20: 5, 21: 0, 22: 336, 23: 11, 24: 2, 25: 0, 26: 14, 27: 22, 28: 1, 29: 6, 30: 27, 31: 0, 32: 0, 33: 1, 34: 0, 35: 69, 36: 427, 37: 883, 38: 9, 39: 52, 40: 1, 41: 0, 42: 4, 43: 8, 44: 1, 45: 15, 46: 0, 47: 81, 48: 0, 49: 90, 50: 86, 51: 120, 52: 1, 53: 4, 54: 276, 55: 8}
