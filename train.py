from optparse import OptionParser
import trainer_ocr

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="train.txt", help="Text file name for training")
parser.add_option("--validationFileName", action="store", type="string", dest="validationFileName", default="val.txt", help="Text file name for validation")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="test.txt", help="Text file name for testing")
parser.add_option("--charsetFileName", action="store", type="string", dest="charsetFileName", default="charset_size=91.txt", help="Charset file containg the character mappings")
# parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=16, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=64, help="Image height for feeding into the network")
# parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in image for feeding into the network")
parser.add_option("--sequentialFetch", action="store_true", dest="sequentialFetch", default=False, help="Sequentially fetch images for each batch")
parser.add_option("--randomFetchTest", action="store_true", dest="randomFetchTest", default=False, help="Randomly fetch images for each test batch")
# parser.add_option("--computeMeanImage", action="store_false", dest="computeMeanImage", default=False, help="Compute mean image on data")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=5, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size (Can only be 1 due to inconsistent image sizes)")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=10000000, help="Progress evaluation step")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./model-inc_res_v2/", help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="inc_res_v2_book", help="Name to be used for saving the model")
parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="./outputImages", help="Directory for saving output images")

# Network Params
parser.add_option("--maxSequenceLength", action="store", type="int", dest="maxSequenceLength", default=70, help="Maximum sequence length")
parser.add_option("--networkStride", action="store", type="int", dest="networkStride", default=16, help="Network stride")
parser.add_option("--numLayersRNN", action="store", type="int", dest="numLayersRNN", default=1, help="Number of layers in the encoder and the decoder")
parser.add_option("--attentionSize", action="store", type="int", dest="attentionSize", default=256, help="Number of units in the attention module (0 corresponds no attention)")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

trainer_ocr.trainModel(options)