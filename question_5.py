import pandas as pd
from tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Initialize data (Question 5)
    columns = ['Id', 'Size', 'Color', 'Shape', 'Usable']
    data = [
        [1, 'medium', 'blue', 'brick', 'Yes'],
        [2, 'small', 'red', 'sphere', 'Yes'],
        [3, 'large', 'green', 'pillar', 'Yes'],
        [4, 'large', 'green', 'sphere', 'Yes'],
        [5, 'small', 'red', 'wedge', 'No'],
        [6, 'large', 'red', 'wedge', 'No'],
        [7, 'large', 'red', 'pillar', 'No']
    ]
    data = pd.DataFrame(data=data, columns=columns)

    # Build the decision tree
    clf = DecisionTreeClassifier(max_depth=10, criterion='entropy')

    # Train
    feature_cols = columns[1:4]
    target_col = columns[4]
    features = data[feature_cols]
    targets = data[target_col]
    clf.fit(features, targets, verbose=1)

    # Check
    predictions = clf.predict(features)
    accuracy = accuracy_score(targets, predictions)
    print('Model accuracy : ', accuracy)
    