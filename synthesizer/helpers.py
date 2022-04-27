from base64 import encode
import spacy
from spacy.matcher import Matcher
import hashlib
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


nlp = spacy.load("en_core_web_sm")

def get_patterns(df, labels, chosenpatterns=None):

    patterns = []


    cols = df.columns.tolist()
    for i in range(3, df.shape[1]):
        temp = dict()
        prf = precision_recall_fscore_support(df["labels"], df.iloc[:, i], average="binary" ) 
        temp["pattern"] = cols[i]
        temp["precision"] = prf[0]
        temp["recall"] = prf[1]
        temp["fscore"] = prf[2]
        patterns.append(temp)
    patterns.sort(key=lambda x: (x["fscore"], len(x["pattern"])), reverse=True)
    
    return patterns

def dict_hash(dictionary):
    dhash = hashlib.md5()

    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def expand_working_list(pat):
    result = []
    if (pat == None):
        return []
    else:
        working_list = []
        optionals = []
        combinations = pat.split("+")
        for pattern in combinations:
            patterns_within = pattern.split("|")
            if (len(patterns_within) > 1):
                # result.append(working_list)
                # working_list = []
                optional_patterns = []
                for p in patterns_within:
                    if (p[0] == "["):
                        temp = {"LEMMA": {"IN": [p[1:-1]]}, "OP": "+"}
                        optional_patterns.append(temp)
                    elif (p[0] == "$"):
                        temp = {"ENT_TYPE": p[1:], "OP": "+"}
                        optional_patterns.append(temp)
                    else:
                        temp = {"POS": p, "OP": "+"}
                        optional_patterns.append(temp)
                count = len(working_list)
                if (count):
                    while (count):
                        count -= 1
                        temp = working_list.pop(0)
                        for opt in optional_patterns:
                            updated_pattern = temp + [opt]
                            working_list.append(updated_pattern)
                else:
                    working_list = [[x] for x in optional_patterns]

                # result.append(optional_patterns)
            else:
                if (pattern == "*"):
                    temp = {"OP": "?"}
                    # temp = {"OP": "?"}
                elif (pattern[0] == "["):
                    temp = {"LEMMA": {"IN": [pattern[1:-1]]}, "OP": "+"}
                elif (pattern[0] == "$"):
                    temp = {"ENT_TYPE": pattern[1:], "OP": "+"}
                else:
                    temp = {"POS": pattern, "OP": "+"}
                if (len(working_list) == 0):
                    working_list.append([temp])
                else:
                    for i in range(len(working_list)):
                        working_list[i].append(temp)
        return working_list


def match_positives(working_list, positive_examples, negative_set=False):
    if (positive_examples == None or len(positive_examples) == 0):
        return 0
    match_count = 0
    matched_sentences = 0
    matcher = Matcher(nlp.vocab)
    for index, distinct_pattern in enumerate(working_list):
        matcher.add(f"Posmatch{index}", [distinct_pattern])
    match_collector = dict()
    for _i, sent in enumerate(positive_examples):
        matched = False
        doc = nlp(sent)
        matches = matcher(doc)
        if (matches is not None and len(matches) > 0):
            for id, start, end in matches:
                if (str(doc[start:end]).strip() != ""):
                    matched = True
                    if (doc.vocab.strings[id] in match_collector):
                        match_collector[doc.vocab.strings[id]].append((start, end))
                    else:
                        match_collector[doc.vocab.strings[id]] = [(start, end)]

                    # print(f'sent={_i} pat={[doc.vocab.strings[id]]}, mathcedspan={doc[start:end]}')
        if (matched):
            matched_sentences += 1
            # print(f"{sent}, {match_collector}")
    if (len(set(match_collector.keys())) == len(working_list)) or negative_set:
        match_count = matched_sentences
    return match_count


def show_patters(patterns, out=None):
    results = sorted(patterns.keys(), key=len, reverse=True)
    print("------writing patterns-----")
    file = out if out != None else "patterns"
    out_file = open(f'../out/{out}', 'w')
    out_file.writelines("---------Matched Patterns--------")
    out_file.writelines("\n")
    for pattern in results:
        out_file.writelines(f"{pattern}, precision = {patterns[pattern][0]}, recall = {patterns[pattern][1]}")
        out_file.writelines("\n")


