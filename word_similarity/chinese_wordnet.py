# chinese_wordnet.py

import re
from CwnGraph import CwnImage

def generate_sub_synset():
    # Load the latest Chinese WordNet Image
    cwn = CwnImage.latest()

    # Get all synsets
    synsets = cwn.get_all_synsets()

    # Pattern to extract synsets
    pattern = r'(?<=\().+(?=，[a-zA-Z])'

    # Extract synsets
    synset = {}
    for i in range(len(synsets)):
        syn = []
        for j in range(len(synsets[i].relations)):
            syn.extend(re.findall(pattern, str(synsets[i].relations[j][1])))
            synset.update({i: syn})

    # Get sub synset
    sub_synset = list(synset.values())

    # Remove non-character strings (Chinese characters)
    sub_synset = [sublist for sublist in sub_synset 
                  if all(re.search(r'^[\u4100-\u9fff]', word) for word in sublist)]

    # Filter sub_synsets with 4-6 words
    sub_synset = [sublist for sublist in sub_synset if 3 < len(sublist) < 7]

    return sub_synset

# Generate sub_synset variable when the module is imported
sub_synset = generate_sub_synset()
