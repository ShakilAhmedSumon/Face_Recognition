from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from random import choice
from argparse import ArgumentParser
import sys

class FaceRecognition():
	def __init__(self,image_file_name):
		self.model = load_model('facenet_keras.h5')
		self.image_file_name = image_file_name


	def extract_face(self,filename, required_size=(160, 160)):
		image = Image.open(filename)
		image = image.convert('RGB')
		pixels = asarray(image)
		detector = MTCNN()
		results = detector.detect_faces(pixels)
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)
		return face_array

	def load_faces(self,directory):
		faces = list()
		for filename in listdir(directory):
			path = directory + filename
			face = self.extract_face(path)
			faces.append(face)
		return faces

	def load_dataset(self,directory):
		X, y = list(), list()
		for subdir in listdir(directory):
			path = directory + subdir + '/'
			if not isdir(path):
				continue
			faces = self.load_faces(path)
			labels = [subdir for _ in range(len(faces))]
			X.extend(faces)
			y.extend(labels)
		return asarray(X), asarray(y)

	def split_data(self):
		trainX, trainy = self.load_dataset('data/train/')
		print(trainX.shape)
		print(trainy.shape)
		savez_compressed('data.npz', trainX, trainy)
		return trainX, trainy

	def load_data_from_npy(self):
		data = load('data.npz')
		trainX, trainy = data['arr_0'], data['arr_1']
		return trainX, trainy


	def get_embedding(self,model, face_pixels):
		face_pixels = face_pixels.astype('float32')
		mean, std = face_pixels.mean(), face_pixels.std()
		face_pixels = (face_pixels - mean) / std
		samples = expand_dims(face_pixels, axis=0)
		yhat = model.predict(samples)
		return yhat[0]

	def get_train_embedding(self):
		trainX, trainy= self.split_data()
		print(trainX.shape)
		print(trainy.shape)
		
		newTrainX = list()
		for face_pixels in trainX:
			embedding = self.get_embedding(self.model, face_pixels)
			newTrainX.append(embedding)
		newTrainX = asarray(newTrainX)
		print("trainX:",newTrainX.shape)
		return newTrainX

	def normalize_train_test(self):
		in_encoder = Normalizer(norm='l2')
		trainX = self.get_train_embedding()
		trainX = in_encoder.transform(trainX)
		return trainX


	def label_encode(self):
		out_encoder = preprocessing.LabelEncoder()
		_, trainy = self.split_data()
		out_encoder.fit(trainy)
		trainy = out_encoder.transform(trainy)
		return trainy

	def train_model(self):
		trainX = self.normalize_train_test()
		trainy = self.label_encode()
		
		model = SVC(kernel='linear', probability=True)
		model.fit(trainX, trainy)
		yhat_train = model.predict(trainX)

		score_train = accuracy_score(trainy, yhat_train)
		print('Accuracy: train=%.3f' % (score_train*100))

		data = load('data.npz')
		trainy = data['arr_1']

		test_image_array = self.extract_face(self.image_file_name)
		test_embedding = self.get_embedding(self.model, test_image_array)

		out_encoder = preprocessing.LabelEncoder()
		out_encoder.fit(trainy)
		samples = expand_dims(test_embedding, axis=0)
		yhat_class = model.predict(samples)
		yhat_prob = model.predict_proba(samples)

		class_index = yhat_class[0]
		class_probability = yhat_prob[0,class_index] * 100
		predict_names = out_encoder.inverse_transform(yhat_class)
		print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

		pyplot.imshow(test_image_array)
		title = '%s (%.3f)' % (predict_names[0], class_probability)
		pyplot.title(title)
		pyplot.show()

	

if __name__ == '__main__':
	image = sys.argv[1]
	print(image)
	fc = FaceRecognition(image)
	fc.train_model()

	
	
	