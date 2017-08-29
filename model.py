import tensorflow as tf

# Import Inception v4 Model
from inception_resnet_v2 import *
import inputReader

def createModel(inputs, maximumSeqLength=100, isTrain=True, imageHeight=64, networkOutputStride=16, numLayersRNN=1, attentionSize=1):
	incResV2_checkpointFile = './inception_resnet_v2_2016_08_30.ckpt'
	if not os.path.isfile(incResV2_checkpointFile):
		# Download file from the link
		import wget
		import tarfile
		url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
		filename = wget.download(url)

		# Extract the tar file
		tar = tarfile.open(filename)
		tar.extractall()
		tar.close()

	# Create model
	arg_scope = inception_resnet_v2_arg_scope()
	with slim.arg_scope(arg_scope):
		net, end_points = inception_resnet_v2_base(inputs, output_stride=networkOutputStride, align_feature_maps=True)
	print ("CNN output dimensions: %s" % str(net.get_shape()))
	# Squeeze the height dimension in the tensor
	# net = tf.squeeze(net, axis=1)
	net = tf.reduce_mean(net, axis=1)
	print ("CNN output dimensions (after squeeze): %s" % str(net.get_shape()))

	# Attach the sequence to sequence model on top of the CNN
	cell = tf.contrib.rnn.BasicLSTMCell # instance of RNNCell

	if numLayersRNN > 1:
		cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(numLayersRNN)])

	if attentionSize != 0:
		assert(attentionSize > 0)
		attentionMechanism = tf.contrib.seq2seq.LuongAttention(512, encoder_outputs)
		cell = tf.contrib.seq2seq.AttentionWrapper(cell, attentionMechanism, attention_size=attentionSize)

	
	if isTrain:
		helper = tf.contrib.seq2seq.TrainingHelper(input=input_vectors, sequence_length=maximumSeqLength)

		# When feed previous is false, all decoder inputs will be used, otherwise, only the first one
		# outputs, states = embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols,
		# 										embedding_size, output_projection=None, feed_previous=False)
	else:
		helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding, start_tokens=tf.tile([GO_SYMBOL], [batchSize]), end_token=END_SYMBOL)

		# When feed previous is false, all decoder inputs will be used, otherwise, only the first one
		# outputs, states = embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols,
		# 										embedding_size, output_projection=None, feed_previous=False)

	decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell, helper=helper, initial_state=cell.zero_state(batchSize, tf.float32))
	outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=20)