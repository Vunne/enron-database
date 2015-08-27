#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import tree
from sklearn import cross_validation
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

f_train, f_test, l_train, l_test = cross_validation.train_test_split(
	features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(f_train, l_train)

print 'accuracy:', clf.score(f_test, l_test)
print 'labels', l_test

pred = clf.predict(f_test)
print '\npred', pred
true_positives = 0
for i in range(len(pred)):
	if pred[i] == 1.:
		if l_test == 1.0:
			true_positives += 1
print '\ntrue positives', true_positives

# Compute confusion matrix
cm = confusion_matrix(l_test, pred)

print '\nconfusion_matrix:'
print cm

print '\nprecision score'
print sklearn.metrics.precision_score(l_test, pred, average='micro')
print '\nrecall score'
print sklearn.metrics.recall_score(l_test, pred)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#practice
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print '\nprecision score'
print sklearn.metrics.precision_score(true_labels, predictions, average='micro')
print '\nrecall score'
print sklearn.metrics.recall_score(true_labels, predictions)
print '\nconfusion_matrix:'
print confusion_matrix(true_labels, predictions)
