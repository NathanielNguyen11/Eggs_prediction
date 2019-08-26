import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import tensorflow as tf
from keras.layers import Dropout
# fix random seed for reproducibility
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import argparse
from keras.optimizers import SGD
from sklearn.linear_model import LinearRegression
import pickle
import csv
import sys
import os

DATA = r'D:\Downloads\Eggs_prediction-master\dataset\bw_train.csv'
test_DATA = r'D:\Downloads\Eggs_prediction-master\dataset\bw_test.csv'
CHECK = '.\model1'
CHECKh5='.\modelh521'
RESULT ='.\sresult1'
PLOT = 'yes'
def parse_arguments(argv):
	parser = argparse. ArgumentParser()

	parser.add_argument('--dataset', type = str,default=DATA, help='Directory to dataset')
	parser.add_argument('--test_dataset', type = str,default=test_DATA, help='Directory to dataset')
	parser.add_argument('--checkckpt_dir',type = str,default=CHECK, help='checkpoint logs')
	parser.add_argument('--checkh5_dir',type = str,default=CHECKh5, help='checkpoint logs')
	parser.add_argument('--result', type=str, default=RESULT, help='Directory to ref dataset')
	parser.add_argument('--plot',type = str,default=PLOT, help='yes/no')

	return parser.parse_args(argv)
def write_csv(result_dir,refdataset,prediction,dataset_size):
	result = os.path.join(result_dir, 'result.csv')
	with open(result, "w", newline='') as result:
		writer = csv.writer(result)
		writer.writerows(map(lambda x: [x],prediction))

def main(args):
	dataset = args.dataset
	test_data = args.test_dataset
	check_dir = args.checkckpt_dir
	checkh5_dir = args.checkh5_dir
	result_dir = args.result

	if not os.path.isdir(check_dir):
		os.makedirs(check_dir)
	else:
		pass

	if not os.path.isdir(checkh5_dir):
		os.makedirs(checkh5_dir)
	else:
		pass
	if not os.path.isdir(result_dir):
		os.makedirs(result_dir)
	else:
		pass

	seed = 7
	numpy.random.seed(seed)
	# load pima indians dataset
	dataset   = numpy.loadtxt(dataset, delimiter=",", skiprows=1)
	test_data = numpy.loadtxt(test_data, delimiter=",", skiprows=1)
	# split into input (X) and output (Y) variables
	dataset_size = dataset.shape[1] - 1
	X = dataset[:,0:dataset.shape[1]-1]
	Y = dataset[:,dataset.shape[1]-1]
	X_testr = test_data[:,0:test_data.shape[1]]

	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(X)
	rescaledX_test = scaler.fit_transform(X_testr)

	(X_train, X_test, Y_train, Y_test) = train_test_split(rescaledX, Y, test_size=0.1, random_state=seed)
	model = LinearRegression().fit(X_train, Y_train)

	filename = 'finalized_model.sav'
	pickle.dump(model, open(filename, 'wb'))

	# load the model from disk
	loaded_model = pickle.load(open(filename, 'rb'))
	Y_pred = loaded_model.predict(rescaledX_test)
	print (Y_pred)
	print (rescaledX_test.shape[0])
	write_csv(result_dir,test_data, Y_pred,rescaledX_test.shape[0])

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
