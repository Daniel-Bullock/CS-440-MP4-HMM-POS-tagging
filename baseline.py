"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    output = []
    word_map = {}
    tag_map = {}

    for sentence in train:
        for word, tag in sentence:
            if word not in word_map:
                word_map[word] = {}
            if tag not in word_map[word]:
                word_map[word][tag] = 1
            else:
                word_map[word][tag] += 1
            if tag not in tag_map:
                tag_map[tag] = 1
            else:
                tag_map[tag] += 1

    maxTag = None
    maxCount = 0
    for tag, count in tag_map.items():
        if count > maxCount:
            maxTag = tag
            maxCount = count


    for sentence in test:
        temp = []
        for word in sentence:
            if word in word_map:
                temp.append((word, max(word_map[word].keys(), key=lambda x: word_map[word][x])))
            else:
                temp.append((word, maxTag))
        output.append(temp)
    return output
