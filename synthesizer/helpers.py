from base64 import encode
import spacy
import copy
from spacy.matcher import Matcher
import hashlib
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle
import copy


from nltk.corpus import wordnet as wn

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "LM"

# import gensim
# import gensim.downloader
# pretrained_vectors = gensim.downloader.load('word2vec-google-news-300')
# model_name = "word2vec"

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
        # print("pat: {} \n working_list {}".format(pat,working_list))
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
    for _i, doc in enumerate(positive_examples):
        matched = False
        matches = matcher(doc)
        if (matches is not None and len(matches) > 0):
            for id, start, end in matches:
                rule = id
                if (str(doc[start:end]).strip() != ""):
                    matched = True
                    if (rule in match_collector):
                        match_collector[rule].append((start, end))
                    else:
                        match_collector[rule] = [(start, end)]

                    # print(f'sent={_i} pat={[doc.vocab.strings[id]]}, mathcedspan={doc[start:end]}')
        if (matched):
            matched_sentences += 1
            # print(f"{sent}, {match_collector}")
    if (len(set(match_collector.keys())) == len(working_list)) or negative_set:
        match_count = matched_sentences
    return match_count

def find_similar_words(lemmas, examples, threshold, topk_on=False, topk=1, negative_set=False):
    # try:
    #     with open('cache/similar_words_negative_set{}_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(negative_set,len(examples),threshold,topk_on,topk), 'rb') as f:
    #         similar_words = pickle.load(f)
    try:
        with open('cache/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(len(examples),threshold,topk_on,topk), 'rb') as f:
            similar_words = pickle.load(f)
    except FileNotFoundError:
        similar_words = dict()
    alreadyIn = set()
    for lemma in lemmas:
        if lemma in similar_words:
            alreadyIn.add(lemma)
            continue
        print("new lemma")
        similar_words[lemma] = dict()
    if len(alreadyIn) == len(lemmas):
        return similar_words
    for lemma in lemmas:
        if lemma in alreadyIn: continue
        lemma_embeddings = model.encode(lemma, convert_to_tensor=True)
        for _i, ex in enumerate(examples):
            doc = nlp(str(ex))
            for token in doc:
                #get similarity
                if not str(token.lemma_) in similar_words[lemma] and not str(token.lemma_) in lemmas and not token.is_stop and not token.is_punct and len(token.text.strip(" "))>1:
                    similar_words[lemma][token.lemma_] = util.cos_sim(model.encode(token.lemma_, convert_to_tensor=True), lemma_embeddings)[0][0]

                    # similar_words[lemma][token.lemma_] = token.similarity(nlp(lemma))

                    # if token.lemma_ in pretrained_vectors and lemma in pretrained_vectors:
                    #     similar_words[lemma][token.lemma_] = pretrained_vectors.similarity(token.lemma_,lemma)

                    # pos = ''
                    # if token.pos_ == 'NOUN': pos = 'n'
                    # if token.pos_ == 'VERB': pos = 'v'
                    # if token.pos_ == 'ADV': pos = 'r'
                    # if token.pos_ == 'ADJ': pos = 'a'
                    # synset1 = wn.synsets(lemma, pos)
                    # synset2 = wn.synsets(token.lemma_, pos)
                    # if len(synset1) > 0 and len(synset2) > 0:
                    #     if token.lemma_ in similar_words[lemma] and token.pos_ in similar_words[lemma]:
                    #         similar_words[lemma][token.lemma_][token.pos_] = max(synset1[0].wup_similarity(synset2[0]),similar_words[lemma][token.lemma_][0])
                    #     else:
                    #         similar_words[lemma][token.lemma_] = [synset1[0].wup_similarity(synset2[0]), token.pos_]

                    # if token.lemma_ in similar_words[lemma] and token.pos_ in similar_words[lemma][token.lemma_]:
                    #     similar_words[lemma][token.lemma_][token.pos_] = 

    # wordnet
    # for lemma in lemmas:
    #     if lemma in alreadyIn: continue
    #     if not topk_on:
    #         similar_words[lemma] = {k:v for k,v in similar_words[lemma].items() if v[0]>threshold}
    #     else:
    #         similar_words[lemma] = {k:v for k,v in sorted(similar_words[lemma].items(), key=lambda item: item[1][0], reverse=True)[:topk]}

    for lemma in lemmas:
        if lemma in alreadyIn: continue
        if not topk_on:
            similar_words[lemma] = {k:v for k,v in similar_words[lemma].items() if v>threshold}
        else:
            similar_words[lemma] = {k:v for k,v in sorted(similar_words[lemma].items(), key=lambda item: item[1], reverse=True)[:topk]}

    with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),threshold,topk_on,topk), 'wb') as f:
        pickle.dump(similar_words,f)
    return similar_words

def soft_match_positives(working_list, negative_set=False, price=None, similarity_dict=None, threshold=0.6, topk_on=False, topk=1):
    match_count = 0
    matched_sentences = 0
    matcher = Matcher(nlp.vocab)
    lemmas = []
    for index, distinct_pattern in enumerate(working_list):
        for pattern in distinct_pattern:
            if 'LEMMA' in pattern and 'IN' in pattern['LEMMA'] and pattern['OP'] == '+':
                lemmas += pattern['LEMMA']['IN']
        
    lemmas = list(set(lemmas))
    if len(lemmas) > 0:
        # print('start find' + str(len(positive_examples)))
        # similar_words = find_similar_words(lemmas, price["example"].values, threshold,negative_set=negative_set,topk_on=topk_on,topk=topk)
        similar_words = {}
        if not similarity_dict is None:
            similar_words = similarity_dict
        else:
            similar_words = get_similar_words(similarity_dict, price["example"].values, threshold,topk_on=topk_on,topk=topk)
        # similar_words = similarity_dict
        # print(similar_words)

        for lemma in lemmas:
            if type(similar_words[lemma]) == type([]): continue
            similar_words[lemma] = [k for k,v in similar_words[lemma].items()]

        #wordnet   
        # for lemma in lemmas:
        #     similar_words[lemma] = [[k,v[1]] for k,v in similar_words[lemma].items()]
            
        for index, distinct_pattern in enumerate(working_list):
            for pattern in distinct_pattern:
                if 'LEMMA' in pattern and 'IN' in pattern['LEMMA'] and pattern['OP'] == '+':
                    if len(pattern['LEMMA']['IN']) <= 1 and pattern['LEMMA']['IN'][0] in similar_words:
                        pattern['LEMMA']['IN'] += similar_words[pattern['LEMMA']['IN'][0]]
                        # pattern['POS'] = nlp(pattern['LEMMA']['IN'][0])[0].pos_

        #wordnet
        # length = len(working_list)
        # for index, distinct_pattern in enumerate(working_list):
        #     if index >= length: break
        #     for index_j, pattern in enumerate(distinct_pattern):
        #         if 'LEMMA' in pattern and 'IN' in pattern['LEMMA'] and pattern['OP'] == '+':
        #             if pattern['LEMMA']['IN'][0] in similar_words:
        #                 for v in similar_words[pattern['LEMMA']['IN'][0]]:
        #                     temp = copy.deepcopy(working_list[index])
        #                     temp[index_j]['LEMMA']['IN'] = [v[0]]
        #                     # temp[index_j]['POS'] = v[1]
        #                     working_list.append(temp)

        
                # print(pattern)
    #         matcher.add(f"Posmatch{index}", [distinct_pattern])
        
    # else:
    #     # print("no LITERAS")
    #     for index, distinct_pattern in enumerate(working_list):
    #         matcher.add(f"Posmatch{index}", [distinct_pattern])
    # match_collector = dict()
    # for _i, doc in enumerate(positive_examples):
    #     matched = False
    #     matches = matcher(doc)
    #     if (matches is not None and len(matches) > 0):
    #         for id, start, end in matches:
    #             rule = id
    #             if (str(doc[start:end]).strip() != ""):
    #                 matched = True
    #                 if (rule in match_collector):
    #                     match_collector[rule].append((start, end))
    #                 else:
    #                     match_collector[rule] = [(start, end)]

    #                 # print(f'sent={_i} pat={[doc.vocab.strings[id]]}, mathcedspan={doc[start:end]}')
    #     if (matched):
    #         matched_sentences += 1
    #         # print(f"{sent}, {match_collector}")
    # if (len(set(match_collector.keys())) == len(working_list)) or negative_set:
    #     match_count = matched_sentences
    # return match_count

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


def get_similarity_dict(examples, num_threshold=2, soft_threshold=0.6, soft_topk_on=False, soft_topk=1):
    try:
        with open('cache/{}/words_dict_examplenum{}.pkl'.format(model_name,len(examples)), 'rb') as f:
            words_dict = pickle.load(f)
        with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),soft_threshold,soft_topk_on,soft_topk), 'rb') as f:
            similar_words = pickle.load(f)
        return words_dict, similar_words 
    except FileNotFoundError:
        try:
            with open('cache/{}/similarity_dict_examplenum{}.pkl'.format(model_name,len(examples)), 'rb') as f:
                similarity_dict = pickle.load(f)
        except FileNotFoundError:
            similarity_dict = dict()    
            words_dict = dict()
            for _i, ex in enumerate(examples):
                doc = nlp(str(ex))
                for token in doc:
                    if not token.is_stop and not token.is_punct and len(token.text.strip(" "))>1:
                        if not str(token.lemma_) in words_dict:
                            words_dict[str(token.lemma_)] = 1
                            similarity_dict[str(token.lemma_)] = {}
                        else: words_dict[str(token.lemma_)] += 1
            words_list = list(words_dict.keys())
            print(len(words_list))
            for _i in range(len(words_list)):
                print("{} / {}".format(_i,len(words_list)))
                for _j in range(_i+1,len(words_list)):
                    if model_name == "word2vec":
                        # i_embeddings = model.encode(words_list[_i], convert_to_tensor=True)
                        if words_list[_i] in pretrained_vectors and words_list[_j] in pretrained_vectors:
                            similarity = pretrained_vectors.similarity(words_list[_i],words_list[_j])
                            # similarity = util.cos_sim(model.encode(words_list[_j], convert_to_tensor=True), i_embeddings)[0][0]
                            similarity_dict[words_list[_i]][words_list[_j]] = similarity
                            similarity_dict[words_list[_j]][words_list[_i]] = similarity
                    if model_name == "LM":
                        i_embeddings = model.encode(words_list[_i], convert_to_tensor=True)
                        # if words_list[_i] in pretrained_vectors and words_list[_j] in pretrained_vectors:
                            # similarity = pretrained_vectors.similarity(words_list[_i],words_list[_j])
                        similarity = util.cos_sim(model.encode(words_list[_j], convert_to_tensor=True), i_embeddings)[0][0]
                        similarity_dict[words_list[_i]][words_list[_j]] = similarity
                        similarity_dict[words_list[_j]][words_list[_i]] = similarity
                        

                    
                    # similar_words[lemma][token.lemma_] = token.similarity(nlp(lemma))

                    # pos = ''
                    # if token.pos_ == 'NOUN': pos = 'n'
                    # if token.pos_ == 'VERB': pos = 'v'
                    # if token.pos_ == 'ADV': pos = 'r'
                    # if token.pos_ == 'ADJ': pos = 'a'
                    # synset1 = wn.synsets(lemma, pos)
                    # synset2 = wn.synsets(token.lemma_, pos)
                    # if len(synset1) > 0 and len(synset2) > 0:
                    #     if token.lemma_ in similar_words[lemma] and token.pos_ in similar_words[lemma]:
                    #         similar_words[lemma][token.lemma_][token.pos_] = max(synset1[0].wup_similarity(synset2[0]),similar_words[lemma][token.lemma_][0])
                    #     else:
                    #         similar_words[lemma][token.lemma_] = [synset1[0].wup_similarity(synset2[0]), token.pos_]

                    # if token.lemma_ in similar_words[lemma] and token.pos_ in similar_words[lemma][token.lemma_]:
                        #     similar_words[lemma][token.lemma_][token.pos_] = 

            with open('cache/{}/similarity_dict_examplenum{}.pkl'.format(model_name,len(examples)), 'wb') as f:
                pickle.dump(similarity_dict,f)
            with open('cache/{}/words_dict_examplenum{}.pkl'.format(model_name,len(examples)), 'wb') as f:
                pickle.dump(words_dict,f)
            
    for lemma in similarity_dict:
        if not soft_topk_on:
            similarity_dict[lemma] = {k:v for k,v in similarity_dict[lemma].items() if v>soft_threshold}
        else:
            similarity_dict[lemma] = {k:v for k,v in sorted(similarity_dict[lemma].items(), key=lambda item: item[1], reverse=True)[:soft_topk]}
    with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),soft_threshold,soft_topk_on,soft_topk), 'wb') as f:
        pickle.dump(similarity_dict,f)
    return words_dict, similarity_dict

def get_similar_words(similarity_dict, examples, threshold, topk_on=False, topk=1):
    try:
        with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),threshold,topk_on,topk), 'rb') as f:
            similar_words = pickle.load(f)
    except FileNotFoundError:
        similar_words = similarity_dict
        words_list = list(similarity_dict.keys())
        for lemma in words_list:
            if not topk_on:
                similar_words[lemma] = {k:v for k,v in similar_words[lemma].items() if v>threshold}
            else:
                similar_words[lemma] = {k:v for k,v in sorted(similar_words[lemma].items(), key=lambda item: item[1], reverse=True)[:topk]}

        with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),threshold,topk_on,topk), 'wb') as f:
            pickle.dump(similar_words,f)
    return similar_words
