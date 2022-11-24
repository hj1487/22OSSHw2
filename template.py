#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/hj1487

import sys
import pandas as pd

def load_dataset(dataset_path):
	data=pd.read_csv(dataset_path)
	data_df=pd.DataFrame(data)
	#load the csv file at the given path into the pandas DataFrame and return the DataFrame
	return data_df

def dataset_stat(dataset_df):	
	#For the given DataFrame, return the following statistical analysis results in order
	#Number of fetures, Number of data for class 0
	#Number of data for class 1
	n_feats=data_df.shape[1]

	n_class0=n_class1=0
	if(data_df['target']==0):
		n_class0=n_class0+1
	elif(data_df['target']==1):
		n_class1=n_class1+1

	return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	#Splitting the given DataFrame and return train data, test data, train label, and test label in order
	#You must split the data using the given test size


	return

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#Using the given train dataset, train the decision tree model-implement with default arguments
	#After training, evaluate the performances of the model using the given test dataset
	#Return three performance metrics (accuracy, precision, recall) in order
	return

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#Using the given train dataset, train the random forest model-default arguments
	#After traiing, evaluate the performances of the model using the given test dataset
	#Return three performance metrics(accuracy, preciison, recall) in order
	return

def svm_train_test(x_train, x_test, y_train, y_test):
	#trian the pipeline consists of a standard scaler and SVM-default argument
	#evaluate the performances of the model using the given test dataset
	#Return three performance metrics in order
	return

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
