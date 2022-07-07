import spacy
from spacy.matcher import Matcher
from synthesizer.pattern_types import *
from synthesizer.helpers import expand_working_list
from synthesizer.helpers import match_positives
from synthesizer.helpers import show_patters
from synthesizer.helpers import soft_match_positives

nlp = spacy.load("en_core_web_sm")

class Synthesizer:
    def __init__(self, positive_examples, negative_examples=None, threshold=0.5,literal_threshold=4, max_depth=10, price=None) -> None:
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
        
    def read_examples(self, file):
        examples =[]
        if(file == None):
            return []
        with open(file) as f:
            lines = f.readlines()
            examples = [line.lower() for line in lines]
        return examples
    
    def get_literals_space(self, threshold=4):
        literal_dict = dict()
        words = []
        for ex in self.positive_examples:
            doc = self.nlp(ex)
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

        return list(literal_dict.keys())[:threshold]       #pick the top N most common words

    def get_search_space(self, literal_threshold=4):
        part_of_speech = [ "PRON","VERB", "PROPN", "NOUN", "ADJ", "ADV", "AUX", "NUM"]
        entities =[ 'DATE', 'EVENT', 'LOC', 'MONEY', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY']
        wild_card = ["*"]
        literals = self.get_literals_space( threshold=20)
        
        
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
            self.search_space.append(symbol)
        for pattern in entities:
            symbol = stru(ENTITY, f"${pattern}")
            self.search_space.append(symbol)
        return self.search_space
    
    
    def search(self, pat,  previous_positive_matched=0, previous_negative_matched=0, depth=0, make_or=False, search_space=None):
        self.search_track.add(pat)
        if(depth>self.max_depth):
            print("***ERROR MAX DEPTH REACHED")
            print(f'Pattern: {pat}')

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
                working_pattern = f"{pat}|{p.value_1}"
                # print("Expanding pattern with or -------- ",working_pattern)
            else:
                if(pat != ""):
                    working_pattern = f"{pat}+{p.value_1}"
                else:
                    working_pattern = f"{p.value_1}"

            #if wildcard is added no need to check. Just go on in the recurssion
            if(p.type_ == WILD):
                self.search( working_pattern,  previous_positive_matched=previous_positive_matched, previous_negative_matched=previous_negative_matched, depth=depth+1, search_space=new_search_space)
                continue

            
            working_list = expand_working_list(working_pattern)
            # print(working_list)

            #to turn on soft match
            soft_match_positives(working_list, self.positive_examples, price=self.price,topk_on=True, topk=3)

            postive_match_count = match_positives(working_list, self.positive_examples)
            negative_match_count = match_positives(working_list, self.negative_examples, negative_set=True)

            #Get penalities for negative example matched and rewards for positive examples matched and decide to prune the branch
            # reward = (postive_match_count - previous_positive_matched)/len(positive_examples)
            try:

                # reward = postive_match_count/(len(self.positive_examples)-previous_positive_matched)
                reward = (postive_match_count-previous_positive_matched)/postive_match_count
                # reward = ((postive_match_count-previous_positive_matched)/postive_match_count)*len()
            except:
                reward = 0
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
            if(reward==0 or penality>0.3):
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
                    self.search(working_pattern,  previous_positive_matched=postive_match_count, previous_negative_matched=negative_match_count, depth=depth+1, search_space=new_search_space, make_or=True)

            else:
                if(make_or):
                    print("Stopping here ", working_pattern, previous_positive_matched, postive_match_count)
                    continue
                if(postive_match_count==0):
                    #No need to go on
                    continue
                if(postive_match_count==previous_positive_matched):
                    if(pat!="" ):
                        print(f"pat {pat} with {p.value_1}")
                        
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
