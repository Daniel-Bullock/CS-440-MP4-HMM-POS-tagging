"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

from collections import Counter

import math as np


def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    output = []
    laplace = 0.00001  # laplace smoothing constant

    tag_pairs = Counter()  # (tag a, tag b), no. of occurrence
    tags_solo = Counter()  # no. of words with tag T
    single_words_count = Counter()  # no. of words w
    words_n_tags = Counter()  # (word w, tag T), no. of occurrence

    total_words = 0
    # print(train)
    for sentence in train:
        prev_tag = ""
        for word, tag in sentence:
            total_words += 1
            if prev_tag != "":
                tag_pair = (prev_tag, tag)
                tag_pairs[tag_pair] += 1

            tags_solo[tag] += 1
            words_n_tags[(word, tag)] += 1
            single_words_count[word] += 1

            prev_tag = tag

    unique_tags_count = len(tags_solo)
    unique_words_count = len(single_words_count)

    for sentence in test:  # inference
        predicted_sentence = []
        trellis = []
        for i in range(0, len(sentence)):
            col = []
            for curr_tag in tags_solo:
                col.append([0, curr_tag, (-1, -1)])
            trellis.append(col)

        curr_word_index = 0
        for each_word in sentence:
            if curr_word_index == 0:
                for tag_idx in range(0, unique_tags_count):
                    curr_tag = trellis[0][tag_idx][1]

                    p_tag = np.log((tags_solo[curr_tag] + laplace) / (total_words + laplace * (unique_words_count + 1)))
                    prob_word = np.log(
                        (words_n_tags[(each_word, curr_tag)] + laplace) / (
                                    tags_solo[curr_tag] + laplace * (unique_words_count + 1)))
                    trellis[0][tag_idx][0] = p_tag + prob_word
            else:
                for tag_idx in range(0, unique_tags_count):
                    curr_tag = trellis[curr_word_index][tag_idx][1]
                    #max_path = np.inf
                    max_path = float("-inf")
                    unique_tags_count_idx = 0

                    for prev_tag_idx in range(0, unique_tags_count):
                        prev_tag = trellis[curr_word_index - 1][prev_tag_idx][1]

                        prob_tag = np.log(
                            (tag_pairs[(prev_tag, curr_tag)] + laplace) / (
                                        tags_solo[prev_tag] + laplace * unique_tags_count))

                        prob_word = np.log(
                            (words_n_tags[(each_word, curr_tag)] + laplace) / (
                                    tags_solo[curr_tag] + laplace * (unique_words_count + 1)))

                        tot_path = prob_tag + prob_word + trellis[curr_word_index - 1][prev_tag_idx][0]

                        if tot_path >= max_path:
                            max_path = tot_path
                            unique_tags_count_idx = prev_tag_idx

                    trellis[curr_word_index][tag_idx][0] = max_path
                    trellis[curr_word_index][tag_idx][2] = (curr_word_index - 1, unique_tags_count_idx)

            curr_word_index += 1

        max_val = float("-inf")
        max_i = 0
        for t_tag in range(0, unique_tags_count):
            curr_final_val = trellis[len(sentence) - 1][t_tag][0]
            if curr_final_val >= max_val:
                max_val = curr_final_val
                max_i = t_tag

        count = len(sentence) - 1
        spot = trellis[count][max_i]
        while count >= 0:
            predicted_sentence.append((sentence[count], spot[1]))
            prev = spot[2]
            #print(prev)
            spot = trellis[prev[0]][prev[1]]
            count -= 1

        predicted_sentence.reverse()
        output.append(predicted_sentence)

    # print(trellis)
    # print(output)
    return output
