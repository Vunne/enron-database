#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data 

import matplotlib.pyplot as plt
import numpy as np 
        
#########################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
print 'Task 1'
finance_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
features_list = ['poi','from_poi', 'to_poi', 'shared_receipt_with_poi'] # You will need to use more features
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


#########################################################################
### Task 2: Remove outliers
print 'Task 2'
# remove the entry 'TOTAL' from data_dict because it is the sum of all the other entries. It should not be in data_dict
data_dict.pop('TOTAL', 0)
# remove "THE TRAVEL AGENCY IN THE PARK" because it is not a person
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
# remove "LOCKHART EUGENE E" because it's all NaN (no data)
data_dict.pop("LOCKHART EUGENE E", 0)
# remove "BHATNAGAR SANJAY" because his negative restricted_stock
data_dict.pop("BHATNAGAR SANJAY", 0)


#########################################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print 'Task 3'
my_dataset = data_dict

# create "to_poi" and "from_poi" ratios for email features
for name in my_dataset:
	if my_dataset[name]['from_this_person_to_poi'] == 'NaN': f_poi = 0.
	else: f_poi = my_dataset[name]['from_this_person_to_poi'] * 1.
	if my_dataset[name]['from_messages'] == 'NaN': total = 0.
	else: total = my_dataset[name]['from_messages']
	if total == 0.: my_dataset[name]['from_poi'] = 0.
	else: my_dataset[name]['from_poi'] = f_poi / total
	
	if my_dataset[name]['from_poi_to_this_person'] == 'NaN': f_poi = 0.
	else: f_poi = my_dataset[name]['from_poi_to_this_person'] * 1.
	if my_dataset[name]['to_messages'] == 'NaN': total = 0.
	else: total = my_dataset[name]['to_messages']
	if total == 0.: my_dataset[name]['to_poi'] = 0.
	else: my_dataset[name]['to_poi'] = f_poi / total

f_poi_NoPOI_data = []
t_poi_NoPOI_data = []
f_poi_data = []
t_poi_data = []
for name in my_dataset:
	if my_dataset[name]['poi'] == False:
		f_poi_NoPOI_data.append(my_dataset[name][features_list[1]])
		t_poi_NoPOI_data.append(my_dataset[name][features_list[2]])
	else:
		f_poi_data.append(my_dataset[name][features_list[1]])
		t_poi_data.append(my_dataset[name][features_list[2]])
plt.scatter(f_poi_data, t_poi_data, color="c")
plt.scatter(f_poi_NoPOI_data, t_poi_NoPOI_data, color="r")
plt.xlabel("from poi")
plt.ylabel("to poi")
# plt.show()

# Create np.array of the financial features to perform a PCA of them
# the result are called financial_pc1 and financial_pc2
from sklearn.preprocessing import MinMaxScaler

finance_data = []
for e in finance_features:
	data_to_scale = []	#create a list to scale each finance feature before dumping it in finance_data
	for name in my_dataset:
		if my_dataset[name][e] != 'NaN':
			data_to_scale.append( my_dataset[name][e] * 1.)
		else:
			data_to_scale.append( 0.0)
	data_to_scale = np.array([[i] for i in data_to_scale])
	scaler = MinMaxScaler()
	scaled_data = scaler.fit_transform(data_to_scale)
	for i, name2 in zip(range(len(scaled_data)), my_dataset):
		my_dataset[name2][e] = scaled_data[i][0]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

finance_data = featureFormat(my_dataset, finance_features, sort_keys = True)
finance_labels, finance_features = targetFeatureSplit(data)



########################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
print 'Task 4'

from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import preprocessing
features = preprocessing.scale(features)

# PCA and Pipeline
pca = RandomizedPCA(n_components=2)

#clf = Pipeline(steps=[('pca', pca), ('dt', DecisionTreeClassifier(min_samples_leaf=1, random_state=42))])
#clf = Pipeline(steps=[('pca', pca), ('svr', svm.SVC())])


print '\nStarting GridSearchCV...'
params_svm = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
#clf = Pipeline([('pca', pca), ('gridCV', GridSearchCV(svr, params_svm, n_jobs=5, pre_dispatch=1))])

params_dt = {'min_samples_leaf':[1, 5], 'random_state':[35, 42, 50]}
dt = DecisionTreeClassifier()
clf = Pipeline([('pca', pca), ('gridCV', GridSearchCV(dt, params_dt, n_jobs=5, pre_dispatch=1))])
print '\nGridSearchCV finished\n', clf

pca.fit(finance_features)
print '\npca.explained_variance_ratio_', pca.explained_variance_ratio_, '\n'

print 'CLF', clf

#print '\nBest estimator:', clf.best_estimator_

# extraction of components to plot them
print '\npca.explained_variance_ratio_', pca.explained_variance_ratio_, '\n'
financial_pc1 = pca.components_[0]
financial_pc2 = pca.components_[1]

transformed_data = pca.transform(features)
for ii, jj in zip(transformed_data, features):
	plt.scatter( financial_pc1[0]*ii[0], financial_pc1[1]*ii[0], color='r')
	plt.scatter( financial_pc2[0]*ii[0], financial_pc2[1]*ii[0], color='c')
	plt.scatter( jj[0], jj[1], color="b")

plt.xlabel("bonus")
plt.ylabel("long-term incentive")
plt.show()

clf.fit(features, labels)

########################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print 'Testing classifier...\n'
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)