import csv
import sys
import numpy as np
import re
import os
import nltk
import json
from params import *

TRAIN_LABELS_CSV = "dataset/labels/train/labels.train.csv"
DEV_LABELS_CSV = "dataset/labels/dev/labels.dev.csv"

TRAIN_EXAMPLES_DIR = "dataset/speech_transcriptions/train/tokenized/"
DEV_EXAMPLES_DIR = "dataset/speech_transcriptions/dev/tokenized/"

TRAIN_IVECTORS_FILE = "dataset/ivectors/train/ivectors.json"
DEV_IVECTORS_FILE = "dataset/ivectors/dev/ivectors.json"

GLOVE_PATH = "wordVecs/glove.6B.50d.txt"
GLOVE_DIM = 50

UNIVERSAL_TAGSET = nltk.tag.mapping._UNIVERSAL_TAGS
UNIVERSAL_TAGSET_INDICES = { tag : UNIVERSAL_TAGSET.index(tag) + GLOVE_DIM for tag in UNIVERSAL_TAGSET }

FEATURE_DIM = GLOVE_DIM + len(UNIVERSAL_TAGSET)

def load_train_ivectors():
    with open(TRAIN_IVECTORS_FILE, 'r') as json_file:
        data = json.load(json_file)

    N = len(data.keys())
    ivectors = np.array([ data[key] for key in sorted(data.keys()) ], dtype=np.float32)

    return ivectors

def load_dev_ivectors():
    with open(DEV_IVECTORS_FILE, 'r') as json_file:
        data = json.load(json_file)

    N = len(data.keys())
    ivectors = np.array([ data[key] for key in sorted(data.keys()) ], dtype=np.float32)

    return ivectors

def load_train_labels():
    return parse_labels(TRAIN_LABELS_CSV) 

def load_dev_labels():
    return parse_labels(DEV_LABELS_CSV)

def get_languages():
    return ["ARA", "CHI", "FRE", "GER", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR"]

def parse_labels(filepath):
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
    
    return labels

def load_glove_dict(glove_path=GLOVE_PATH):
    f = open(glove_path, "r")
    vocab = {}
    for line in f:
        row = line.strip().split()
        vocab[row[0]] = np.array([ float(x) for x in row[1:] ], dtype=np.float32)
    f.close()
    return vocab

def load_train_examples(glove_dict):
    examples, lengths = load_examples(TRAIN_EXAMPLES_DIR, glove_dict)
    return examples, lengths

def load_dev_examples(glove_dict):
    examples, lengths = load_examples(DEV_EXAMPLES_DIR, glove_dict)
    return examples, lengths

def load_examples(example_dir, glove_dict):
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
