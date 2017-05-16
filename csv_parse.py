import csv
import sys

def parse_labels(filepath):
    """
    Reads in the label CSV file at filepath.

    Returns: an dictionary of speaker ID to L1 classification
    """

    languages = get_languages() 

    csvfile = open(filepath, 'r')
    reader = csv.reader(csvfile)
    next(reader, None) # skip the header row
    labels = {}
    for row in reader:
        speaker_id, speech_prompt, essay_id, L1 = row
        labels[int(speaker_id)] = languages.index(L1)
    csvfile.close()

    return labels

def get_train_labels():
    train_path = "data/nli-shared-task-2017/data/labels/train/labels.train.csv"
    return parse_labels(train_path) 

def get_dev_labels():
    dev_path = "data/nli-shared-task-2017/data/labels/dev/labels.dev.csv"
    return parse_labels(dev_path)

def get_languages():
    return ["ARA", "CHI", "FRE", "GER", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR"]

if __name__ == "__main__":
    print "{} training labels".format(len(get_train_labels()))
    print "{} dev labels".format(len(get_dev_labels()))
    print "Languages: {}".format(" ".join(get_languages()))
