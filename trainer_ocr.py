import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import datetime as dt
import cPickle as pkl

# Import custom data
import input_reader
import model

def trainModel(options):
	inputReader = input_reader.InputReader(options)

	# Data placeholders
	# Image height is fixed while the system automatically adjusts the image width
	with tf.variable_scope('Model'):
		inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, options.imageHeight, None, 3], name="inputBatchImages")
		inputBatchLabels = tf.placeholder(dtype=tf.float32, shape=[None, options.maxSequenceLength, inputReader.vocabSize], name="inputBatchLabels")
		inputKeepProbability = tf.placeholder(dtype=tf.float32, name="inputKeepProbability")

		scaledInputBatchImages = tf.scalar_mul((1.0/255), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

	predictions = model.createModel(scaledInputBatchImages, maximumSeqLength=options.maxSequenceLength, isTrain=options.trainModel, 
									imageHeight=options.imageHeight, networkOutputStride=options.networkStride, 
									numLayersRNN=options.numLayersRNN, attentionSize=options.attentionSize):

	# Create list of vars to restore before train op
	variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"])

	with tf.name_scope('Loss'):
		# Define loss
		weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))
		sequence_loss = tf.contrib.seq2seq.sequence_loss(predictions, inputBatchLabels, weights=weights)

		tf.losses.add_loss(sequence_loss)
		# tf.losses.add_loss(cross_entropy_loss_aux_logits)
		loss = tf.reduce_mean(tf.losses.get_losses())

	with tf.name_scope('Accuracy'):
		correct_predictions = tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(inputBatchLabels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

	with tf.name_scope('Optimizer'):
		# Define Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

		# Op to calculate every variable gradient
		gradients = tf.gradients(loss, tf.trainable_variables())
		gradients = list(zip(gradients, tf.trainable_variables()))

		# Op to update all variables according to their gradient
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

	# Initializing the variables
	init = tf.global_variables_initializer()
	init_local = tf.local_variables_initializer()

	if options.tensorboardVisualization:
		# Create a summary to monitor cost tensor
		tf.summary.scalar("loss", loss)

		# Create summaries to visualize weights
		for var in tf.trainable_variables():
		    tf.summary.histogram(var.name, var)
		# Summarize all gradients
		for grad, var in gradients:
		    tf.summary.histogram(var.name + '/gradient', grad)

		# Merge all summaries into a single op
		mergedSummaryOp = tf.summary.merge_all()

	# 'Saver' op to save and restore all the variables
	saver = tf.train.Saver()

	bestLoss = 1e9
	step = 1

	# Train model
	if options.trainModel:
		with tf.Session() as sess:
			# Initialize all variables
			sess.run(init)
			sess.run(init_local)

			if options.startTrainingFromScratch:
				print ("Removing previous checkpoints and logs")
				os.system("rm -rf " + options.logsDir)
				os.system("rm -rf " + options.modelDir)
				os.system("mkdir " + options.modelDir)

				# Load the pre-trained Inception ResNet v2 model
				restorer = tf.train.Saver(variables_to_restore)
				restorer.restore(sess, inc_res_v2_checkpoint_file)

			# Restore checkpoint
			else:
				print ("Restoring from checkpoint")
				saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
				saver.restore(sess, options.modelDir + options.modelName)

			if options.tensorboardVisualization:
				# Op for writing logs to Tensorboard
				summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

			print ("Starting network training")
			try:
				# Keep training until reach max iterations
				while True:
					batchImagesTrain, batchLabelsTrain = inputReader.getTrainBatch()
					# print ("Batch images shape: %s, Batch labels shape: %s" % (batchImagesTrain.shape, batchLabelsTrain.shape))

					# If training iterations completed
					if batchImagesTrain is None:
						print ("Training completed")
						break

					# Run optimization op (backprop)
					if options.tensorboardVisualization:
						[trainLoss, currentAcc, _, summary] = sess.run([loss, accuracy, applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
						# Write logs at every iteration
						summaryWriter.add_summary(summary, step)
					else:
						[trainLoss, currentAcc, _] = sess.run([loss, accuracy, applyGradients], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})

					print ("Iteration: %d, Minibatch Loss: %f, Accuracy: %f" % (step, trainLoss, currentAcc * 100))
					step += 1

					if step % options.saveStep == 0:
						# Save model weights to disk
						saver.save(sess, options.modelDir + options.modelName)
						print ("Model saved: %s" % (options.modelDir + options.modelName))

					#Check the accuracy on test data
					if step % options.evaluateStep == 0:
						# Report loss on test data
						batchImagesTest, batchLabelsTest = inputReader.getTestBatch()

						[testLoss, testAcc] = sess.run([loss, accuracy], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
						print ("Test loss: %f, Test Accuracy: %f" % (testLoss, testAcc))

						# #Check the accuracy on test data
						# if step % options.saveStepBest == 0:
						# 	# Report loss on test data
						# 	batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
						# 	[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
						# 	print ("Test loss: %f" % testLoss)

						# 	# If its the best loss achieved so far, save the model
						# 	if testLoss < bestLoss:
						# 		bestLoss = testLoss
						# 		# bestModelSaver.save(sess, best_checkpoint_dir + 'checkpoint.data')
						# 		bestModelSaver.save(sess, checkpointPrefix, global_step=0, latest_filename=checkpointStateName)
						# 		print ("Best model saved in file: %s" % checkpointPrefix)
						# 	else:
						# 		print ("Previous best accuracy: %f" % bestLoss)

			except KeyboardInterrupt:
				print("Process interrupted by user") # Save the model and exit

			# Save final model weights to disk
			saver.save(sess, options.modelDir + options.modelName)
			print ("Model saved: %s" % (options.modelDir + options.modelName))

			# Report loss on test data
			batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
			testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
			print ("Test loss (current): %f" % testLoss)

			print ("Optimization Finished!")

	# Test model
	if options.testModel:
		print ("Testing saved model")

		os.system("rm -rf " + options.imagesOutputDirectory)
		os.system("mkdir " + options.imagesOutputDirectory)

		# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
		with tf.Session() as session:
			saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
			saver.restore(session, options.modelDir + options.modelName)

			# Get reference to placeholders
			predictionsNode = session.graph.get_tensor_by_name("Predictions:0")
			accuracyNode = session.graph.get_tensor_by_name("Accuracy/accuracy:0")
			inputBatchImages = session.graph.get_tensor_by_name("Model/inputBatchImages:0")
			inputBatchLabels = session.graph.get_tensor_by_name("Model/inputBatchLabels:0")
			inputKeepProbability = session.graph.get_tensor_by_name("Model/inputKeepProbability:0")

			inputReader.resetTestBatchIndex()
			accumulatedAccuracy = 0.0
			numBatches = 0
			fileNames = []
			predictedLabels = []
			while True:
				batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
				files = inputReader.getFileNames(isTrain=False)
				if batchLabelsTest is None:
					break
			
				[predictions, accuracy] = session.run([predictionsNode, accuracyNode], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
				# print ('Current test batch accuracy: %f' % (accuracy))
				accumulatedAccuracy += accuracy

				fileNames = fileNames + files
				predictedLabels = predictedLabels + np.argmax(predictions, axis=1).tolist()

				numBatches += 1

				# Save image results
				inputReader.saveLastBatchResults(batchImagesTest, predictedLabels, isTrain=False)

				if numBatches == 20:
					break

			with open('results.npy', 'wb') as fp:
				pkl.dump(fileNames, fp)
				pkl.dump(predictedLabels, fp)

			with open("output.txt", "w") as file:
				for i in range(len(fileNames)):
					file.write(fileNames[i] + " " + str(predictedLabels[i]) + "\n")

		accumulatedAccuracy = accumulatedAccuracy / numBatches
		print ('Cummulative test set accuracy: %f' % (accumulatedAccuracy * 100))

		print ("Model tested")
