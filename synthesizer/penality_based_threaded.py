from symtable import Symbol
import spacy
from nltk.corpus import wordnet
from spacy.matcher import Matcher
from synthesizer.pattern_types import *
from synthesizer.helpers import expand_working_list
from synthesizer.helpers import match_positives
from synthesizer.helpers import soft_match_positives
from synthesizer.helpers import show_patters


nlp = spacy.load("en_core_web_sm")

class Synthesizer:
    def __init__(self, positive_examples, negative_examples=None, threshold=0.5,literal_threshold=4, max_depth=10, rewardThreshold=0.01, penalityThreshold=0.3, price=None, soft_match_on=False, words_dict=None, similarity_dict=None, soft_threshold=0.6, soft_topk_on=False, soft_topk=1) -> None:
        self.soft_match_on = soft_match_on
        self.words_dict = words_dict
        self.similarity_dict = similarity_dict
        self.soft_threshold = soft_threshold
        self.soft_topk_on = soft_topk_on
        self.soft_topk = soft_topk

        self.nlp = spacy.load("en_core_web_sm")
        self.threshold = threshold
        self.patterns_set= dict()
        self.max_depth = max_depth
        self.positive_examples = self.read_examples(positive_examples) if type(positive_examples)==type(str()) else positive_examples
        self.negative_examples = self.read_examples(negative_examples) if type(negative_examples)==type(str()) else negative_examples
        self.search_space = [] 
        self.get_search_space(literal_threshold)
        self.candidate = []
        self.patterns_set = {}
        self.search_track = set()
        self.price = price

        self.rewardThreshold = rewardThreshold
        self.penalityThreshold = penalityThreshold
        
        
    
    def read_examples(self, file):
        examples =[]
        if(file == None):
            return []
        with open(file) as f:
            lines = f.readlines()
            examples = [line.lower() for line in lines]
        return examples
    
    def get_literals_space(self, threshold=4, mention_threshold=3):
        literal_dict = dict()
        words = []
        if self.soft_match_on:
            for ex in self.positive_examples:
                doc = self.nlp(str(ex))
                for token in doc:
                    # print(token)
                    if not token.is_stop and not token.is_punct and len(token.text.strip(" "))>1 and not token.pos_ == "NUM" and str(token.lemma_) in self.similarity_dict:
                        lit = str(token.lemma_)
                        if(not lit in literal_dict):
                            literal_dict[lit] = 1
                        else:
                            literal_dict[lit] += 1
                        words.append(token.lemma_)
            simi_dict = dict()
            for word in words:
                simi_dict[str(word)] = literal_dict[str(word)]
                for sim_words in self.similarity_dict[word]:
                    if sim_words in words: simi_dict[str(word)] += literal_dict[str(sim_words)]

            # literal_dict =  {k: v for k, v in sorted(literal_dict.items(), key=lambda item: item[1], reverse=True)}
            simi_dict =  {k: v for k, v in sorted(simi_dict.items(), key=lambda item: item[1], reverse=True)}
            simi_list = list(simi_dict.keys())
            print(simi_dict)
            final_list = []
            # final_list = simi_list[:threshold]

            for lit in simi_list:
                flag = True
                for word in final_list:
                    if lit in self.similarity_dict[word]:
                        flag = False
                        break 
                if flag:
                    final_list.append(lit)
                    if len(final_list) >= threshold: break
                
            return final_list
        else:
            for ex in self.positive_examples:
                print(type(ex))
                doc = self.nlp(str(ex))
                for token in doc:
                    # print(token)
                    if not token.is_stop and not token.is_punct and len(token.text.strip(" "))>1:
                        lit = str(token.lemma_)
                        if(not lit in literal_dict):
                            literal_dict[lit] = 1
                        else:
                            literal_dict[lit] += 1
                        words.append(token.lemma_)

            literal_dict =  {k: v for k, v in sorted(literal_dict.items(), key=lambda item: item[1], reverse=True)}

            literal_space = []

            for word, count in literal_dict.items():
                if(count<mention_threshold):
                    break
                else:
                    literal_space.append(word)

            return literal_space
    
    def get_synonyms(self, literals, threshold=1):
        synonyms_ls = []
        for pattern in literals:
            pattern = nlp(pattern)
            synonyms_dict = dict()
            for ex in self.positive_examples:
                doc = self.nlp(str(ex))
                for token in doc:
                    if not str(token.lemma_) in literals and not str(token.lemma_) in synonyms_ls and not token.is_stop and not token.is_punct and len(token.text.strip(" "))>1:
                        synonyms_dict[str(token.lemma_)] = token.similarity(pattern)
            synonyms_dict = {k: v for k, v in sorted(synonyms_dict.items(), key=lambda item: item[1], reverse=True)}
            synonyms_ls += list(synonyms_dict.keys())[:threshold]
        return synonyms_ls

    def get_search_space(self, literal_threshold=4):
        part_of_speech = [ "PRON","VERB", "PROPN", "NOUN", "ADJ", "ADV", "AUX", "NUM"]
        entities =[ 'DATE', 'EVENT', 'LOC', 'MONEY', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY']
        wild_card = ["*"]
        literals = self.get_literals_space( literal_threshold)
        
        
            #All POS tags
        for pattern in part_of_speech:
            symbol = stru(POS, pattern)
            self.search_space.append(symbol)
            #All WildCard tags
        for pattern in wild_card:
            symbol = stru(WILD, pattern)
            self.search_space.append(symbol)
            #All literal word tags
        for pattern in literals:
            symbol = stru(LITERAL, f"[{pattern}]")
            print("literal: " + symbol.type_ + symbol.value_1)
            self.search_space.append(symbol)
            symbol = stru(LITERAL, f"({pattern})")
            print("soft literal: " + symbol.type_ + symbol.value_1)
            self.search_space.append(symbol)

        for pattern in entities:
            symbol = stru(ENTITY, f"${pattern}")
            self.search_space.append(symbol)
        return self.search_space
    
    
    def search(self, pat,  previous_positive_matched=0, previous_negative_matched=0, depth=0, make_or=False, search_space=None):
        self.search_track.add(pat)
        if(depth>=self.max_depth):
            # print("***ERROR MAX DEPTH REACHED")
            # print(f'Pattern: {pat}')
            recall = previous_positive_matched / len(self.positive_examples)
            try:
                precision = previous_positive_matched/(previous_positive_matched+previous_negative_matched)
            except:
                print("Error caught - ", pat, previous_negative_matched, previous_positive_matched)
            fscore = 2*(recall*precision)/(recall+precision)

            self.patterns_set[pat] = [precision, recall, fscore]
            return

        for p in search_space:
            if(depth==0 and p.type_==WILD):
                continue
            new_search_space = search_space[:]
            new_search_space.remove(p)

            if(make_or):
                if(p.type_==WILD):
                    continue
                if pat.rstrip(pat.rsplit('+',1)[-1])+p.value_1+"|"+pat.rsplit('+',1)[-1] in self.search_track:
                    continue
                if self.soft_match_on and p.type_ == LITERAL and (pat.rsplit('+',1)[-1].startswith("(") or pat.rsplit('+',1)[-1].startswith("[")) and pat.rsplit('+',1)[-1].strip("()[]") == p.value_1.strip("()[]"):
                    continue
                working_pattern = f"{pat}|{p.value_1}"
                
                
            else:
                if(pat != ""):
                    working_pattern = f"{pat}+{p.value_1}"
                else:
                    working_pattern = f"{p.value_1}"
            
            #if wildcard is added no need to check. Just go on in the recurssion
            if(p.type_ == WILD):
                self.search( working_pattern,  previous_positive_matched=previous_positive_matched, previous_negative_matched=previous_negative_matched, depth=depth+1, search_space=new_search_space)
                continue

            
            working_list = expand_working_list(working_pattern, soft_match_on=self.soft_match_on, similarity_dict=self.similarity_dict)
            # print(working_list)

            #to turn on soft match
            # soft_match_positives(working_list, self.positive_examples, price=self.price, threshold=0.6)
            soft_match_positives(working_list, price=self.price, similarity_dict=self.similarity_dict, threshold=self.soft_threshold)
            
            postive_match_count = match_positives(working_list, self.positive_examples)
            negative_match_count = match_positives(working_list, self.negative_examples, negative_set=True)

            #Get penalities for negative example matched and rewards for positive examples matched and decide to prune the branch
            # reward = (postive_match_count - previous_positive_matched)/len(positive_examples)
            try:

                reward = postive_match_count/(len(self.positive_examples)-previous_positive_matched)
            except:
                reward = postive_match_count/len(self.positive_examples)
            try:
            # penality = (previous_negative_matched - negative_match_count)/len(negative_examples)
                penality = negative_match_count/len(self.negative_examples)
            except:
                penality = 0

            recall = previous_positive_matched / len(self.positive_examples)

            try:
                precision = previous_positive_matched/(previous_positive_matched+previous_negative_matched)
            except:
                precision = 0
            try:
                fscore = 2*(recall*precision)/(recall+precision)
            except:
                fscore = 0

            if(reward<=self.rewardThreshold or penality>self.penalityThreshold):
                #We know that the previous pattern was working because it got this far without being pruned so we add to the list of candidates
                if(len(pat)>2 and pat[-1]=="*"):
                    # patterns_set.add(pat[:-2])
                    if(len(pat[:-2])>1):
                        self.patterns_set[pat[:-2]] = [precision, recall, fscore]

                else:
                    if(len(pat)>1):
                        self.patterns_set[pat] =  [precision, recall, fscore]
                continue



            if(postive_match_count>previous_positive_matched and depth<self.max_depth):
                self.search(working_pattern, previous_positive_matched=postive_match_count, previous_negative_matched=negative_match_count, depth=depth+1, search_space=new_search_space)
                if(previous_positive_matched==0 and postive_match_count<len(self.positive_examples)):
                    #Search with an or too
                    pass
                self.search(working_pattern,  previous_positive_matched=postive_match_count, previous_negative_matched=negative_match_count, depth=depth+1, search_space=new_search_space, make_or=True)

            else:
                if(make_or):
                    # print("Stopping here ", working_pattern, previous_positive_matched, postive_match_count)
                    continue
                if(postive_match_count==0):
                    #No need to go on
                    continue
                if(postive_match_count==previous_positive_matched):
                    if(pat!="" ):
                        # print(f"pat {pat} with {p.value_1}")
                        
                        self.search(working_pattern, previous_positive_matched=postive_match_count, previous_negative_matched=negative_match_count, depth=depth+1, search_space=new_search_space)
                else:
                    if(pat!="" and not make_or):
                        pass
                    #try an or
                        self.search(working_pattern,  previous_positive_matched=postive_match_count, previous_negative_matched=negative_match_count, depth=depth+1, make_or=True, search_space=new_search_space)

                if(postive_match_count>=len(self.positive_examples)//2):
                    # print(f"pattern oring at {pat} with {working_pattern} matched {postive_match_count} previous match {previous_positive_matched}")
                    self.patterns_set[pat] =  [precision, recall, fscore]
                    self.search(working_pattern,  previous_positive_matched=postive_match_count, previous_negative_matched=negative_match_count, depth=depth+1, make_or=True, search_space=new_search_space)


    def find_patters(self, outfile="out", ):

        search_space = self.search_space
        
        self.search("",  search_space=search_space)
        results = sorted(self.patterns_set,key=len, reverse=True)
        
        
        return results


    

    





# def main():
#     synthh = Synthesizer(positive_examples = "../examples/price_big", negative_examples = "../examples/small_neg")
#     synthh.find_patters(outfile="small_thresh")

# if __name__ == "__main__":
#     main()
