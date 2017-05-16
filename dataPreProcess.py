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
    embed.append(DASH_embed)
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
This function maps an example of words into word indices
@ rawInput: should be in 2D dimension of (numExample, sentences)
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
This function pad all the examples into max length 
"""
def pad2MaxLength(inputsInIndex, maxLength):
    paddedInputsInIndex = []
    for i in range(len(inputsInIndex)):
        inputLen = len(inputsInIndex[i])
        padLen = maxLength - inputLen
        paddedInput = inputsInIndex[i] + [PAD_index] * padLen
        paddedInputsInIndex.append(paddedInput)
    return paddedInputsInIndex

"""
This function loads transcript training set in text,
and generate a training set that represents each word
as word index. All the examples are padded to the same 
length as the max transcript length among all subjects. 
----------------------------------------------------------------
@ testSetFileDir: the file directory of training set transcripts
@ vocab: a dict to map words to id
"""
def makeTrainSet(testSetFileDir,vocab, flag):
    examples = [] 
    count = 1
    # 2. for each batch convert the example into word index and find the
    #    max length of this batch using mapContex2VocabIndex
    # 3. pad to same length 
    for root, dirs, files in os.walk(testSetFileDir):
        for file in files:
            fp = open(os.path.join(root, file), "r")
            context = ' '.join(fp.read().splitlines())
            examples.append(context)
            fp.close()
            count += 1
            if flag == "Small" and count == 10: break
    return examples


def mapVocab2Vec(examplesM, embed):
    # 4. map each word index to its real word vector from glove
    # input: 2D list, row is each example, col is length of each example
    # load glove with above method loadGloVec()
    # save into memory 3D matrix (N, L, W_50); pickle.dump()
    N = len(examplesM)
    L = len(examplesM[0])
    W = 50
    wordToVec = np.zeros((N, L, W))
    for n in range(N):
        for l in range(L):
            word = examplesM[n][l]
            gloVec = embed[word] # 50 long glove?
            wordToVec[n][l] = gloVec

    f = open('wordToVec.pckl','wb')
    pickle.dump(wordToVec,f)
    f.close()

def load_train_examples():
    with open('wordToVec.pckl', 'rb') as handle:
        examples = pickle.load(handle)
    return examples

"""
This function should randomly choose N examples and 
make a 3D batch (N * L * W_50) for the neural network 
to train
------------------------------------------------------

@return: a 3D numpy matrix with shape (N * L * W_50)
"""
#def genBatch(n):
    # 1. Seperate data into batches (n examples/batch) and 
    #    make the 3D Matrix of each batch 
    # return np.zeroes(N, L, 50)
#    pass 

## ==== TEST FUNCTIONS ==== ##
"""
This function tests the mapping of word to index and 
makes sure they are mapped correctly
"""
"""
def testMapVocab2Index(rawInput,inputInIndex,vocab):
    reversedVocab = {v:k for k,v in vocab.iteritems()}
    for i in range(len(rawInput)):
       print "Testing input: ", rawInput[i]
	   restoredInput = [reversedVocab[index] for index in inputInIndex[i]]
	   print restoredInput
"""	

if __name__ == "__main__":
    vocab, embed = loadGloVec(vocabFileDir)
    with open('vocab.pckl', 'rb') as handle:
        vocab = pickle.load(handle)
    examples =  makeTrainSet("dataset/speech_transcriptions/train/tokenized",vocab,"Small")
    examplesInIndex, maxLength = mapContex2VocabIndex(examples, vocab)
    paddedInputsInIndex = pad2MaxLength(examplesInIndex,maxLength)
    mapVocab2Vec(paddedInputsInIndex, embed)
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