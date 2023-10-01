import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Condition:
	def __init__(self, col, val):
		'''
			Splits a pandas data frame based on a condition applied on one of
			its column. For numeric columns, the condition will match when the
			value of that column is greater than or equal to self.val

			Args :
				col : Column name in the dataframe
				val : The split value. equality condition is applied when the column
					  is categorical. greater than or equal to condition will be applied
					  if the column is numeric
		'''
		self.col = col 
		self.val = val 
		self.impurity = 0.0
		self.str = ""

	def __str__(self):
		return self.str

	def _is_numeric(self, x):
		'''
			Checks if a value or a np.ndarray is of numeric type
			Args :
				x : A scalar or a np.ndarray vector
		'''
		if(not isinstance(x, np.ndarray)):
			return isinstance(x, int) or isinstance(x, float)
		return np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.number)
	
	def match(self, data):
		'''
			Returns a boolean mask which is True for rows that matches the condition of this
			object, False otherwise/

			Args :
				data : A pd.DataFrame to match against the condition
		'''
		col = data[self.col]

		if(self._is_numeric(col)):
			self.str = f'{col} >= {self.val} ?'
			return col >= self.val
		else:
			return col == self.val

	def split(self, data):
		'''
			Returns two index Series of the rows that match and do not match the condition
			Args :
				data : A pd.DataFrame to be splitted
		'''
		mask = self.match(data)

		true_split = mask[mask == True]
		false_split = mask[mask == False]

		return true_split.index, false_split.index

class Node:
	def __init__(self, condition, true_branch, false_branch):
		'''
			Non-leaf node in the decision tree.

			Args :
				condition : A Condition object that helps splitting data into 
				either branch.
				true_branch : Pointer to the child Node where the rows satisfy
				the condition held by the Condition object.
				false_branch : Pointer to the child Node where the rows do not
				satisfy the condition held by the condition object.
		'''
		self.condition = condition 
		self.true_branch = true_branch 
		self.false_branch = false_branch

class Leaf:
	def __init__(self, classes):
		'''
			Leaf node in the decision tree.
			Args :
				classes : Classes of the rows that are splitted into this leaf 
				node. Prediction is created based on the class that makes up the 
				majority.
		'''
		if(isinstance(classes, pd.Series)):
			classes = classes.values 
		class_counts = Counter(classes)

		self.classes = classes 
		self.prediction = class_counts.most_common(1)[0][0]


# Decision tree classifier - without tree pruning
class DecisionTreeClassifier:
	def __init__(self, max_depth=10, criterion='gini', numeric_split_strategy='decile'):
		'''
			Classification decision tree.

			Args :
				max_depth : The maximum height of the decision tree
				criterion : The splitting criterion
		'''
		self.criterion_name = criterion
		self.numeric_split_strategy = numeric_split_strategy
		self.max_depth = max_depth
		self.root = None

		self.criterion = self._get_gini_index
		if(criterion != 'gini'):
			if(criterion == 'entropy'):
				self.criterion = self._get_entropy
			elif(criterion == 'classification_error'):
				self.criterion = self._get_classification_error
			else:
				raise Exception('Invalid splitting criterion')

	def _get_unique_values(self, class_arr):
		return np.unique(class_arr)

	def _is_numeric(self, x):
		return np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.number)

	def _get_class_freq(self, classes):
		unique, counts = np.unique(classes.copy(), return_counts=True)

		return {x:y for x, y in zip(unique, counts)}

	# Criterion based on Gini impurity
	def _get_gini_index(self, classes):
		gini = 1
		for _class in self._get_unique_values(classes):
			p = len(classes[classes == _class])/len(classes)
			gini -= p ** 2

		return gini

	# Criterion based on Entropy
	def _get_entropy(self, classes):
		entropy = 0 
		for _class in self._get_unique_values(classes):
			p = len(classes[classes == _class])/len(classes)
			entropy += -p * np.log2(p)

		return entropy

	def _get_classification_error(self, classes):
		p_max = 0
		for _class in self._get_unique_values(classes):
			p = len(classes[classes == _class])/len(classes)
			if(p_max < p):
				p_max = p
		return 1 - p_max

	def _get_best_split(self, data, classes, verbose=0):
		columns = set(data.columns)
		best_impurity = 1
		best_condition = None

		for col in columns:
			values = self._get_unique_values(data[col])

			# If this column is a numeric column, values are set
			# as the deciles of the column instead
			if(self._is_numeric(data[col])):
				if(self.numeric_split_strategy == 'decile'):
					values = [np.percentile(data[col], i*10) for i in range(1, 10)]
				else:
					values = np.unique(data[col])

			for val in values:
				condition = Condition(col, val)
				# Get the split mask for true branch and false branch
				true_split, false_split = condition.split(data)

					# Get the ratios of samples in 2 branches
				true_p = len(true_split) / len(data)
				false_p = len(false_split) / len(data)

				# If we cannot partition based on this condition -> skip
				if(len(true_split) == 0 or len(false_split) == 0): continue

				impurity_true = self.criterion(classes[true_split])
				impurity_false = self.criterion(classes[false_split])
				impurity = impurity_true * true_p + impurity_false * false_p 

				if(impurity < best_impurity):
					best_impurity = impurity 
					best_condition = condition
					best_condition.impurity = impurity

				if(verbose >= 2):
					print('  -- Splitting by attribute ', col, ' value = ', val, ' impurity = ', impurity)

		return best_impurity, best_condition

	# Tree induction function
	def _build_tree(self, data, classes, current_depth=1, verbose=0):
		best_impurity, best_condition = self._get_best_split(data, classes, verbose=verbose)
		
		if(best_condition is None or current_depth >= self.max_depth or len(np.unique(classes)) == 1):
			if(verbose >= 1):
				print('\t'*(current_depth - 1) + f'[*] Reached leaf, current depth {current_depth}, Classes : {self._get_class_freq(classes)}')
			return Leaf(classes)

		print('\t'*(current_depth - 1) + f'[*] Best split found for {best_condition.col}={best_condition.val}, impurity ({self.criterion_name})={best_condition.impurity}')
		true_split, false_split = best_condition.split(data)

		print('\t'*current_depth + '- True branch :')
		true_data = data.loc[true_split]
		true_classes = classes[true_split]
		true_branch = self._build_tree(true_data, true_classes, current_depth=current_depth+1, verbose=verbose)

		print('\t'*current_depth + '- False branch :')
		false_data = data.loc[false_split]
		false_classes = classes[false_split]
		false_branch = self._build_tree(false_data, false_classes, current_depth=current_depth+1, verbose=verbose)

		return Node(best_condition, true_branch, false_branch)

	# Train the classification tree
	def fit(self, data, classes, verbose=0):
		'''
			A wrapper function that calls the tree induction function _build_tree
			to train this decision tree classifier.

			Args:
				data : pd.DataFrame or np.ndarray. A matrix of features used to 
				predict the target classes.
				classes : pd.Series or np.ndarray. Array of classes to predict.
		'''
		self.root = self._build_tree(data, classes, verbose=verbose)

	# Classify function
	def _classify(self, row, node=None):
		if(node is None):
			return None

		if(isinstance(node, Leaf)):
			return node.prediction

		if(node.condition.match(row)):
			return self._classify(row, node.true_branch)
		else:
			return self._classify(row, node.false_branch)

	# A wraper to perform batch classification
	def predict(self, rows):
		'''
			A wrapper function for _classify to perform batch prediction.

			Args :
				rows : pd.DataFrame or np.ndarray. A matrix of features used
				to predict the target classes.
		'''
		labels = []

		for i in range(len(rows)):
			label = self._classify(rows.iloc[i], node=self.root)
			labels.append(label)

		return labels
