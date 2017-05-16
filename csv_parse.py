import csv
import sys
import numpy as np
import pickle

def load_train_labels():
    with open('trainSetlabel.pckl', 'rb') as handle:
        labels = pickle.load(handle)
    return labels

def load_dev_labels():
    with open('devSetlabel.pckl', 'rb') as handle:
        labels = pickle.load(handle)
    return labels

def get_train_labels():
    train_path = "data/nli-shared-task-2017/data/labels/train/labels.train.csv"
    return parse_labels(train_path) 

def get_dev_labels():
    dev_path = "data/nli-shared-task-2017/data/labels/dev/labels.dev.csv"
    return parse_labels(dev_path)

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
    labels = []#{}
    for row in reader:
        speaker_id, speech_prompt, essay_id, L1 = row
        labels.append(languages.index(L1))
    csvfile.close()

    labels = np.array(labels).reshape(len(labels),1)

    f = open(savedName, 'wb')
    pickle.dump(labels,f)
    f.close()

if __name__ == "__main__":
#    print "{} training labels".format(len(get_train_labels()))
#    print "{} dev labels".format(len(get_dev_labels()))
#    print "Languages: {}".format(" ".join(get_languages()))
    train_path = "dataset/labels/train/labels.train.csv"
    parse_labels(train_path,'trainSetlabel.pckl') 
    dev_path = "dataset/labels/dev/labels.dev.csv"
    parse_labels(dev_path,'devSetlabel.pckl')
