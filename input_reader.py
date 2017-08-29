import os
import os.path
import numpy as np
import skimage
import skimage.io
import skimage.transform
import skimage.color
import random
import cv2
from optparse import OptionParser

import charset_utils

class InputReader:
	charset = None
	charset_CharsToNum = None
	vocabSize = 0
	GO_SYMBOL = 0
	END_SYMBOL = 1

	def __init__(self, options):
		self.options = options

		# Reads pathes of images together with their labels
		self.imageList, self.labelList = self.readImageNames(self.options.trainFileName)
		self.imageListVal, self.labelListVal = self.readImageNames(self.options.validationFileName)
		self.imageListTest, self.labelListTest = self.readImageNames(self.options.testFileName)

		self.currentIndex = 0
		self.currentIndexVal = 0
		self.currentIndexTest = 0
		self.totalEpochs = 0
		self.totalImages = len(self.imageList)
		self.totalImagesVal = len(self.imageListVal)
		self.totalImagesTest = len(self.imageListTest)

		# Initialize the charset
		self.charset, self.charset_CharsToNum = charset_utils.readCharset(options.charsetFileName)
		self.vocabSize = len(self.charset)

	def readImageNames(self, imageListFile):
		"""Reads a .txt file containing paths and transcriptions
		Args:
		   imageListFile: a .txt file with one /path/to/image per line along with their corresponding transcription
		Returns:
		   List with all fileNames in file imageListFile along with their corresponding transcription
		"""
		f = open(imageListFile, 'r')
		fileNames = []
		transcriptions = []
		for line in f:
			data = line.strip().split('|')
			fileName = data[0].strip()
			transcription = data[1].strip()
			fileNames.append(fileName)
			transcriptions.append(transcription)

		return fileNames, transcriptions

	def readImagesFromDisk(self, fileNames):
		"""Consumes a list of filenames and returns images
		Args:
		  fileNames: List of image files
		Returns:
		  4-D numpy array: The input images
		"""
		images = []
		for i in range(0, len(fileNames)):
			if self.options.verbose > 1:
				print ("Image: %s" % fileNames[i])

			# Read image (load previous or next one in case of error)
			try:
				img = skimage.io.imread(fileNames[i])
			except KeyboardInterrupt:
				exit()
			except:
				newI = i+1 if i < (len(fileNames) - 1) else i-1
				img = skimage.io.imread(fileNames[newI])

			# Convert image to rgb if grayscale
			if (len(img.shape) == 2) and (self.options.imageChannels == 3):
				img = skimage.color.gray2rgb(img)

			# Rescale image to specific height
			# print ("Previous shape: %s" % str(img.shape))
			# aspectRatio = img.shape[1] / img.shape[0] # Width / height
			# newHeight = self.options.imageHeight
			# newWidth = int(np.ceil(aspectRatio * newHeight))
			# img = skimage.transform.resize(img, (newHeight, newWidth), mode='reflect')

			img = skimage.transform.resize(img, (self.options.imageHeight, self.options.imageWidth), mode='reflect')
			# print ("New shape: %s" % str(img.shape))

			images.append(img)

		# Convert list to ndarray
		images = np.array(images)
		return images

	def getTrainBatch(self):
		"""Returns training images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: training images and labels in batch.
		"""
		if self.totalEpochs >= self.options.trainingEpochs:
			return None, None

		endIndex = self.currentIndex + self.options.batchSize
		if self.options.sequentialFetch:
			# Fetch the next sequence of images
			self.indices = np.arange(self.currentIndex, endIndex)

			if endIndex > self.totalImages:
				# Replace the indices which overshot with 0
				self.indices[self.indices >= self.totalImages] = np.arange(0, np.sum(self.indices >= self.totalImages))
		else:
			# Randomly fetch any images
			self.indices = np.random.choice(self.totalImages, self.options.batchSize)

		imagesBatch = self.readImagesFromDisk([self.imageList[index] for index in self.indices])
		# labelsBatch = self.convertLabelsToOneHot([self.labelList[index] for index in self.indices])
		labelsBatch = self.encodeStrings([self.labelList[index] for index in self.indices])

		self.currentIndex = endIndex
		if self.currentIndex > self.totalImages:
			print ("Training epochs completed: %f" % (self.totalEpochs + (float(self.currentIndex) / self.totalImages)))
			self.currentIndex = self.currentIndex - self.totalImages
			self.totalEpochs = self.totalEpochs + 1

			# Shuffle the image list if not random sampling at each stage
			if self.options.sequentialFetch:
				np.random.shuffle(self.imageList)

		return imagesBatch, labelsBatch

	def resetTestBatchIndex(self):
		self.currentIndexTest = 0

	def getFileNames(self, isTrain=True):
		if isTrain:
			imageList = [self.imageList[index] for index in self.indices]
		else:
			imageList = [self.imageListTest[index] for index in self.indices]

		return imageList

	def getTestBatch(self):
		"""Returns testing images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: test images and labels in batch.
		"""
		# Optional Image and Label Batching
		if self.currentIndexTest >= self.totalImagesTest:
			return None, None
	
		endIndex = self.currentIndexTest + self.options.batchSize
		if endIndex > self.totalImagesTest:
			endIndex = self.totalImagesTest
		self.indices = np.arange(self.currentIndexTest, endIndex)
		imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices])
		# labelsBatch = self.convertLabelsToOneHot([self.labelListTest[index] for index in self.indices])
		labelsBatch = self.encodeStrings([self.labelListTest[index] for index in self.indices])
		self.currentIndexTest = endIndex

		return imagesBatch, labelsBatch

	def restoreCheckpoint(self, numSteps):
		"""Restores current index and epochs using numSteps
		Args:
		  numSteps: Number of batches processed
		Returns:
		  None
		"""
		processedImages = numSteps * self.options.batchSize
		self.totalEpochs = processedImages / self.totalImages
		self.currentIndex = processedImages % self.totalImages

	def convertLabelsToOneHot(self, labels):
		oneHotLabels = []

		for label in labels:
			oneHotVector = np.zeros([self.options.maxSequenceLength, self.vocabSize])
			oneHotVector[len(label)+1:, self.END_SYMBOL] = 1
			intEncoding = [self.END_SYMBOL] * self.options.maxSequenceLength
			intEncoding[0] = self.GO_SYMBOL
			oneHotVector[0, self.GO_SYMBOL] = 1
			for idx in range(len(label)):
				intEncoding[idx+1] = self.charset_CharsToNum[label[idx]]
				oneHotVector[idx+1, intEncoding[idx+1]] = 1

			oneHotLabels.append(oneHotVector)
		oneHotLabels = np.array(oneHotLabels)
		return oneHotLabels

	def encodeStrings(self, labels):
		encodedStrings = []

		for label in labels:
			intEncoding = [self.END_SYMBOL] * self.options.maxSequenceLength
			intEncoding[0] = self.GO_SYMBOL
			for idx in range(len(label)):
				intEncoding[idx+1] = self.charset_CharsToNum[label[idx]]
				
			encodedStrings.append(intEncoding)
		encodedStrings = np.array(encodedStrings)
		return encodedStrings


	def saveLastBatchResults(self, images, predictions, isTrain=True):
		"""Saves the results of last retrieved image batch
		Args:
		  images: 4D Numpy array [batchSize, H, W, numClasses]
		  predictions: List containing the labels
		  isTrain: If the last batch was training batch
		Returns:
		  None
		"""
		labels = ["_iO" if predictions[i] == 0 else "_niO" for i in range(len(predictions))]
		if isTrain:
			imageNames = [self.imageList[index] for index in self.indices]
		else:
			imageNames = [self.imageListTest[index] for index in self.indices]

		images = np.squeeze(images)
		# Iterate over each image name and save the results
		for i in range(self.indices.shape[0]):
			imageName = imageNames[i].split('/')
			imageName = imageName[-1]
			if isTrain:
				imageName = self.options.imagesOutputDirectory + '/' + 'train_' + imageName[:-4] + labels[i] + imageName[-4:]
			else:
				imageName = self.options.imagesOutputDirectory + '/' + 'test_' + imageName[:-4] + labels[i] + imageName[-4:]
			# print(imageName)

			im = images[i, :, :]
			im  = im.astype(np.uint8)
			# minVal = np.min(im)
			# maxVal = np.max(im)
			# scaledIm = (im - minVal) / (maxVal - minVal)
			# scaledIm = scaledIm * 255.0
			# scaledIm = np.uint8(scaledIm)

			skimage.io.imsave(imageName, im)

"""For testing data fetching pipeline independently"""
if __name__ == "__main__":
	# Command line options
	parser = OptionParser()

	parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
	parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="train.txt", help="Text file name for training")
	parser.add_option("--validationFileName", action="store", type="string", dest="validationFileName", default="val.txt", help="Text file name for validation")
	parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="test.txt", help="Text file name for testing")
	parser.add_option("--charsetFileName", action="store", type="string", dest="charsetFileName", default="charset_size=95.txt", help="Charset file containg the character mappings")
	parser.add_option("--sequentialFetch", action="store_true", dest="sequentialFetch", default=False, help="Sequentially fetch images for each batch")
	parser.add_option("--randomFetchTest", action="store_true", dest="randomFetchTest", default=False, help="Randomly fetch images for each test batch")
	parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=5, help="Training epochs")
	parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size")
	parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=512, help="Image Width")
	parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=128, help="Image Height")
	parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Image Channels")
	parser.add_option("--maxSequenceLength", action="store", type="int", dest="maxSequenceLength", default=70, help="Maximum sequence length")
	parser.add_option("--networkStride", action="store", type="int", dest="networkStride", default=16, help="Network stride")

	# Parse command line options
	(options, args) = parser.parse_args()

	inputReader = InputReader(options)
	images, labels = inputReader.getTrainBatch()

	while images is not None:
		# integerEncoding = np.argmax(labels, axis=2)
		# assert (int(np.sum(labels)) == (options.maxSequenceLength))
		integerEncoding = labels
		print ("Images shape: %s" % str(images.shape))
		print ("Labels shape: %s" % str(labels.shape))
		print (integerEncoding)
		print (inputReader.charset)
		strings = charset_utils.convertToString(inputReader.charset, integerEncoding)

		for idx, string in enumerate(strings):
			cv2.imshow("Image", images[idx])
			print ("Transcription: %s" % (string))
			keyPressed = cv2.waitKey(-1)
			if keyPressed == ord('q') or keyPressed == ord('Q'):
				exit(0)
		# Reload the next batch
		images, labels = inputReader.getTrainBatch()