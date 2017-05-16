import csv
import sys

def parse_labels(filepath):
    """
    Reads in the label CSV file at filepath.

    Returns:
        labels - an dictionary of speaker ID to L1 classification
        languages - an array of L1 names
    """

    languages = ["ARA", "CHI", "FRE", "GER", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR"]

    csvfile = open(filepath, 'r')
    reader = csv.reader(csvfile)
    next(reader, None) # skip the header row
    labels = {}
    for row in reader:
        speaker_id, speech_prompt, essay_id, L1 = row
        labels[int(speaker_id)] = languages.index(L1)
    csvfile.close()

    return labels, languages

if __name__ == "__main__":
    print parse_labels(sys.argv[1])
