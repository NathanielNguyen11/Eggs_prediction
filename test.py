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
import csv
from keras.optimizers import SGD
import sys
import os


TDATA = '.\data_test1.csv'
CHECK_DIR = '.\modelh5'
RESULT ='.\sresult2'

def parse_arguments(argv):
	parser = argparse. ArgumentParser()
	parser.add_argument('--testdata', type = str,default=TDATA, help='Directory to dataset')
	parser.add_argument('--check_dir',type = str,default=CHECK_DIR, help='checkpont logs')
	parser.add_argument('--result', type=str, default=RESULT, help='Directory to ref dataset')

	return parser.parse_args(argv)
def load_checkpoint(check_dir):

	json_file = open('%s/model.json'%check_dir, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("%s/model.h5"%check_dir)
	print("Loaded model from disk")
	loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return loaded_model


def write_csv(result_dir,refdataset,prediction):
	index = 0
	result = os.path.join(result_dir, 'result.csv')
	with open(refdataset, "r") as refdataset:
		reader = csv.reader(refdataset)
		next(reader)
		with open(result, "w", newline='') as result:
			writer = csv.writer(result)
			for r in reader:
				r[7] = prediction[index]
				writer.writerows([r])
				index = index + 1

def main(args):
	test_dataset = args.testdata
	check_dir = args.check_dir
	result_dir = args.result

	if not os.path.isdir(result_dir):
		os.makedirs(result_dir)
	else:
		pass

	dataset = numpy.loadtxt(test_dataset, delimiter=",", skiprows=0)
	X_test = dataset[:,3:7]

	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(X_test)
	# load model
	loaded_model = load_checkpoint(check_dir)

	Y_pro = loaded_model.predict(rescaledX)
	Y_predict = loaded_model.predict_classes(rescaledX)

	print(Y_pro)
	print(Y_predict)
	# for i in range (0, len(Y_predict)):
	# 	print (Y_predict[i])

	write_csv(result_dir,test_dataset, Y_predict)
	print('Success in prediction ')
if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
