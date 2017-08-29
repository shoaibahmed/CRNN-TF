"""
	Utils related to character set
	Executing the script directly will create the charset based on the defined vocab
"""

import os
import re
import tensorflow as tf

def readCharset(filename, null_character='\0'):
	"""Reads a charset definition from a tab separated text file.

	charset file has to have format compatible with the FSNS dataset.

	Args:
	filename: a path to the charset file.
	null_character: a unicode character used to replace '<null>' character. the ascii value is \0.

	Returns:
	a dictionary with keys equal to character codes and values - ascii characters.
	"""
	if not os.path.isfile(filename):
		print ("Error: Charset file not found (%s)" % (filename))
		exit(-1)

	pattern = re.compile(r'(\d+)\t(.+)')
	charset = {}
	charsetInv = {}
	with tf.gfile.GFile(filename) as f:
		for i, line in enumerate(f):
			m = pattern.match(line)
			if m is None:
				logging.warning('incorrect charset file. line #%d: %s', i, line)
				continue
			code = int(m.group(1))
			# char = m.group(2).decode('utf-8')
			char = m.group(2)
			if char == '<go>':
				char = ''
			elif char == '<nul>':
				char = null_character
			charset[code] = char
			charsetInv[char] = code

	# charsetCharToNum = dict (zip(charset.values(), charset.keys()))
	return charset, charsetInv

def convertToString(charset, encoding):
	"""
	Converts given integer encoding to string using the provided charset
	"""
	strings = []
	for idx, enc in enumerate(encoding):
		strings.append("")
		for c in enc:
			if c == 1: # End symbol
				break
			strings[idx] += charset[c]

	return strings

def createCharset():
	""" 
	Removed '|' since it serves as the separator for training file and very unlikely to occur in real text
	"""
	vocab = ["<go>", "<nul>", " ", "!", "\\", "/", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", \
			 ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", \
			 "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", \
			 "W", "X", "Y", "Z", "[", "]", "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", \
			 "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "}", "{", "}"]

	vocabSize = len(vocab)
	fileName = "charset_size=" + str(vocabSize) + ".txt"

	with open(fileName, "w") as vocabFile:
		for vocabItemIdx, vocabItem in enumerate(vocab):
			vocabFile.write(str(vocabItemIdx) + "\t" + vocabItem + "\n")
	print ("Charset written to file: %s" % (fileName))

if __name__ == "__main__":
	print ("Creating charset")
	createCharset()
