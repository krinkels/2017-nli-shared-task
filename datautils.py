import csv
import sys
import numpy as np
import cPickle
import re
import os
import nltk
from params import *

TRAIN_LABELS_CSV = "dataset/labels/train/labels.train.csv"
DEV_LABELS_CSV = "dataset/labels/dev/labels.dev.csv"
TRAIN_LABELS_PICKLE = "train_labels.pckl"
DEV_LABELS_PICKLE = "dev_labels.pckl"
TRAIN_EXAMPLES_DIR = "dataset/speech_transcriptions/train/tokenized/"
DEV_EXAMPLES_DIR = "dataset/speech_transcriptions/dev/tokenized/"
TRAIN_EXAMPLES_PICKLE = "train_examples.pckl"
DEV_EXAMPLES_PICKLE = "dev_examples.pckl"
TRAIN_LENS_PICKLE = "train_lens.pckl"
DEV_LENS_PICKLE = "dev_lens.pckl"

GLOVE_PATH = "wordVecs/glove.6B.50d.txt"
GLOVE_DICT_PICKLE = "glove_dict.pckl"
GLOVE_DIM = 50

UNIVERSAL_TAGSET = nltk.tag.mapping._UNIVERSAL_TAGS
UNIVERSAL_TAGSET_INDICES = { tag : UNIVERSAL_TAGSET.index(tag) + GLOVE_DIM for tag in UNIVERSAL_TAGSET }

FEATURE_DIM = GLOVE_DIM + len(UNIVERSAL_TAGSET)

def load_train_labels():
    with open(TRAIN_LABELS_PICKLE, 'rb') as handle:
        labels = cPickle.load(handle)
    return labels

def load_dev_labels():
    with open(DEV_LABELS_PICKLE, 'rb') as handle:
        labels = cPickle.load(handle)
    return labels

def generate_train_labels():
    return parse_labels(TRAIN_LABELS_CSV, TRAIN_LABELS_PICKLE) 

def generate_dev_labels():
    return parse_labels(DEV_LABELS_CSV, DEV_LABELS_PICKLE)

def get_languages():
    return ["ARA", "CHI", "FRE", "GER", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR"]

def parse_labels(filepath, savedName):
    """
    Reads in the label CSV file at filepath.

    Returns: an dictionary of speaker ID to L1 classification
    """

    languages = get_languages() 

    csvfile = open(filepath, 'r')
    reader = csv.reader(csvfile)
    next(reader, None) # skip the header row
    labels = []
    
    for row in reader:
        speaker_id, speech_prompt, essay_id, L1 = row
        labels.append(languages.index(L1))
    csvfile.close()

    labels = np.array(labels)

    if savedName is not None:
        f = open(savedName, 'wb')
        cPickle.dump(labels,f)
        f.close()
    
    return labels

def generate_glove_dict(glove_path=GLOVE_PATH, pickle_path=GLOVE_DICT_PICKLE):
    f = open(glove_path, "r")
    vocab = {}
    for line in f:
        row = line.strip().split()
        vocab[row[0]] = np.array([ float(x) for x in row[1:] ], dtype=np.float32)
    f.close()
    
    vocab_pickle = open(pickle_path, "wb")
    cPickle.dump(vocab, vocab_pickle)
    vocab_pickle.close()
    
    return vocab

def load_glove_dict(pickle_path=GLOVE_DICT_PICKLE):
    with open(pickle_path, 'rb') as handle:
        vocab = cPickle.load(handle)
    return vocab

def generate_train_examples(glove_dict):
    examples, lengths = generate_examples(TRAIN_EXAMPLES_DIR, glove_dict)
    examples_pickle = open(TRAIN_EXAMPLES_PICKLE, 'wb')
    lens_pickle = open(TRAIN_LENS_PICKLE, 'wb')
    cPickle.dump(examples, examples_pickle)
    cPickle.dump(lengths, lens_pickle)
    examples_pickle.close()
    lens_pickle.close()
    
    return examples, lengths

def load_train_examples():
    with open(TRAIN_EXAMPLES_PICKLE, 'rb') as handle:
        examples = cPickle.load(handle)
    with open(TRAIN_LENS_PICKLE, 'rb') as handle:
        lengths = cPickle.load(handle)
        
    return examples, lengths

def generate_dev_examples(glove_dict):
    examples, lengths = generate_examples(DEV_EXAMPLES_DIR, glove_dict)
    examples_pickle = open(DEV_EXAMPLES_PICKLE, 'wb')
    lens_pickle = open(DEV_LENS_PICKLE, 'wb')
    cPickle.dump(examples, examples_pickle)
    cPickle.dump(lengths, lens_pickle)
    examples_pickle.close()
    lens_pickle.close()

def load_dev_examples():
    with open(DEV_EXAMPLES_PICKLE, 'rb') as handle:
        examples = cPickle.load(handle)
    with open(DEV_LENS_PICKLE, 'rb') as handle:
        lengths = cPickle.load(handle)
        
    return examples, lengths

def generate_examples(example_dir, glove_dict):
    vectors = []
    
    for transcript_file in os.listdir(example_dir):
        f = open(example_dir + transcript_file)
        text = f.read()
        tokens = text.split()
        tags = nltk.pos_tag(tokens, tagset='universal')
        
        vector_length = len(tokens) - tokens[1:].count("-")
        vector = np.zeros((vector_length, FEATURE_DIM), dtype=np.float32)
        
        try:
            index = 0
            for i in range(len(tokens)):
                if i + 1 < len(tokens) and tokens[i + 1] == "-":
                    continue # Skip stuttered tokens, just process the dash symbol

                token, tag = tags[i]
                if token.lower() in ["uh", "um", "-"]:
                    tag = "X" # Catch disfluency symbols

                vector[index, :GLOVE_DIM] = glove_dict.get(token.lower(), np.zeros(GLOVE_DIM))
                vector[index, UNIVERSAL_TAGSET_INDICES[tag]] = 1
                index += 1

            vectors.append(vector)
        except:
            print transcript_file
            break
            
    lengths = np.array([vec.shape[0] for vec in vectors], dtype=np.int64)
    L = np.max(lengths)
    N = len(vectors)
    examples = np.zeros((N, L, FEATURE_DIM), dtype=np.float32)
    for i in range(N):
        examples[i, :lengths[i]] = vectors[i]
        
    return examples, lengths

if __name__ == "__main__":
    print "Generating GloVe dict"
    glove_dict = generate_glove_dict()
    print "Generating training examples"
    generate_train_examples(glove_dict)
    print "Generating dev examples"
    generate_dev_examples(glove_dict)
    print "Generating training labels"
    generate_train_labels()
    print "Generating dev labels"
    generate_dev_labels()