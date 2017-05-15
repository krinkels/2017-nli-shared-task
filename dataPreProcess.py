#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to preprocess raw data
    
"""

import tensorflow as tf
import numpy as np
import re
import os
import pickle
from params import *

"""
This function loads the pretrained glovec and
returns the vocab list and embedding 
"""
def loadGloVec(vocabFileDir):
    vocab = {}
    embed = []
    vocab[UNK_token] = 0 # the token for words not in the pretrained gloVect
    embed.append(UNK_embed)
    file = open(vocabFileDir)
    lineIndex = 1
    for line in file.readlines():
        row = line.strip().split(' ')
	    vocab[row[0]] = lineIndex
	    embed.append([float(elem) for elem in row[1:]])
	    lineIndex += 1
    embed.append(PAD_embed)
    f = open('pretrained_embed.pckl', 'wb')
    pickle.dump(embed,f)
    f.close()
    f = open('vocab.pckl','wb')
    pickle.dump(vocab,f)
    f.close()
    print("Loaded GloVec!")
    print("Total vocab numbers: ",len(embed))
    file.close()
    return vocab, embed


"""
This function maps a sentence of words into word indices
@ rawInput: should be in 2D dimension of (numSentences, sentence)
@ vocab: a dict to map words to id
"""
def mapContex2VocabIndex(rawInput,vocab):
    inputInIndex = []
    maxInputLength = 0
    for sentence in rawInput:
	   sentSplitted = re.findall(r"[\w]+|[^\s\w]",sentence.lower())
	   sentInID = [vocab[word] if word in vocab else vocab[UNK_token] for word in sentSplitted]
	   inputInIndex.append(sentInID)
	   if len(sentInID) > maxInputLength: maxInputLength = len(sentInID)
    return inputInIndex, maxInputLength
       
"""
This function loads actual email conext and response,
and generate a training set that represents each word
as word index. Each row of contexs and responses are 
a pair of input and target output for training the 
neural network.
@ testSetFileDir: the file directory of email contex/response
@ vocab: a dict to map words to id
@ flag: "Small" or "All", "Small" for generating a small test set 
        that contains 100 contex/response pair, "All" for generating
        all the contex/response pair from the test set directory.
"""
def makeTrainSet(testSetFileDir,vocab,flag):
    responses = []
    contexs = []
    count = 0     
    for root, dirs, files in os.walk(testSetFileDir):
	for file in files:
        if file.endswith("..context"):
	       fp = open(os.path.join(root, file), "r")
	       contex = ' '.join(fp.read().splitlines())
           fp = open(os.path.join(root,file[0:-9]+"..out"), "r")
	       response = ' '.join(fp.read().splitlines())
   	       if len(contex) > 0 and len(response) > 0: 
		      contexs.append(contex)
		      responses.append(response)
		      count += 1
		   fp.close()
	    if flag == "Small" and count == 100: break
    return responses, contexs

## ==== TEST FUNCTIONS ==== ##
"""
This function tests the mapping of word to index and 
makes sure they are mapped correctly
"""
def testMapVocab2Index(rawInput,inputInIndex,vocab):
    reversedVocab = {v:k for k,v in vocab.iteritems()}
    for i in range(len(rawInput)):
       print "Testing input: ", rawInput[i]
	   restoredInput = [reversedVocab[index] for index in inputInIndex[i]]
	   print restoredInput
	

if __name__ == "__main__":
#    inputTest = ["This is a test.","I hope this works!","Now we need to WERHIWETWEHIU pad the data?"]
    vocab, embed = loadGloVec(vocabFileDir)
#    mappedInput, maxInputLength = mapContex2VocabIndex(inputTest,vocab)
#    testMapVocab2Index(inputTest,mappedInput,vocab)
#    print "The max input length is: ", maxInputLength
#    responses, contexs =  makeTrainSet("../out",vocab,"Small")
#    resID, maxLength_res = mapContex2VocabIndex(responses,vocab)
#    contID, maxLength_cont = mapContex2VocabIndex(contexs,vocab)
#    f = open('maxLenRes.pckl','wb')
#    pickle.dump(maxLength_res,f)
#    f = open('maxLenCont.pckl','wb')
#    pickle.dump(maxLength_cont,f)
#    f = open('resID.pckl','wb')
#    pickle.dump(resID,f)
#    f = open('contID.pckl','wb')
#    pickle.dump(contID,f)
#    f.close()