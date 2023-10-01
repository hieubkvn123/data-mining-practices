import pandas as pd
from tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	# Initialize the data
	data = [
		['Sunny', 75, 70, 'TRUE', 'Play'],
		['Sunny', 80, 90, 'TRUE', 'No Play'],
		['Sunny', 85, 85, 'FALSE', 'No Play'],
		['Sunny', 72, 95, 'TRUE', 'No Play'],
		['Sunny', 69, 70, 'FALSE', 'Play'],
		['Overcast', 72, 90, 'TRUE', 'Play'],
		['Overcast', 83, 78, 'FALSE', 'Play'],
		['Overcast', 64, 65, 'TRUE', 'Play'],
		['Overcast', 81, 75, 'FALSE', 'Play'],
		['Rain', 71, 80, 'TRUE', 'No Play'],
		['Rain', 65, 70, 'TRUE', 'No Play'],
		['Rain', 75, 80, 'FALSE', 'Play'],
		['Rain', 68, 80, 'FALSE', 'Play'],
		['Rain', 70, 96, 'FALSE', 'Play']
	]
	columns = ['Outlook', 'Temp', 'Humidity', 'Windy', 'Class']
	data = pd.DataFrame(data=data, columns=columns)

	# Build the decision tree with gini impurity
	print('[INFO] Building decision tree using gini impurity...')
	clf = DecisionTreeClassifier(max_depth=10, criterion='gini', numeric_split_strategy='all')

	# Train
	feature_cols = columns[0:4]
	target_col = columns[4]
	features = data[feature_cols]
	targets = data[target_col]
	clf.fit(features, targets, verbose=1)

	# Check
	predictions = clf.predict(features)
	accuracy = accuracy_score(targets, predictions)
	print('Model accuracy : ', accuracy)

	# Build the decision tree with classification error
	print('\n\n[INFO] Building decision tree using classification error...')
	clf = DecisionTreeClassifier(max_depth=10, criterion='classification_error', numeric_split_strategy='all')
	clf.fit(features, targets, verbose=1)
	predictions = clf.predict(features)
	accuracy = accuracy_score(targets, predictions)
	print('Model accuracy : ', accuracy)

