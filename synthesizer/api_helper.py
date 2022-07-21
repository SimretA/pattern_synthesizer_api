from cgitb import reset
from re import T
from synthesizer.linear_network import patterns_against_examples, train_linear_mode
from synthesizer.penality_based_threaded import Synthesizer
from synthesizer.helpers import dict_hash
from synthesizer.helpers import get_patterns
from synthesizer.helpers import get_similarity_dict
from synthesizer.helpers import expand_working_list
import pandas as pd
import json
import spacy
import random
import time
import logging

import asyncio

loop = asyncio.get_event_loop()

nlp = spacy.load("en_core_web_sm")
        

class APIHelper:
    def __init__(self):
        self.positive_examples_collector = {}
        self.negative_examples_collector = {}

        self.negative_phrases = []
        self.theme = "price_service_300"
        self.selected_theme = "price"

        # self.data = pd.read_csv(f"examples/df/{self.theme}.csv")
        # self.data['positive'] = self.data['label'].apply(lambda x: x==self.selected_theme)



        self.data = pd.read_csv(f"examples/df/price_service_500.csv", delimiter=",")
        # self.data['positive'] = self.data[self.selected_theme]


        self.priority_unmatch = []
        self.priority_match = []

        self.labels = {}
        self.themes = {}
        self.results = {}


        self.soft_match_on = True
        self.words_dict = {}
        self.similarity_dict = {}
        self.soft_threshold = 0.6
        self.soft_topk_on = False
        self.topk = 1
        self.words_dict, self.similarity_dict = get_similarity_dict(self.data["example"].values, soft_threshold=self.soft_threshold)
        # print(list(self.similarity_dict['pricey'].keys()))

        self.element_to_label = {}
        self.theme_to_element = {}
        self.element_to_sentence = {}
        self.synthesizer_collector = {}
        self.initialize_synthesizers(self.get_themes())


    
    
    def save_cache(self, pattern_set, positive_examples= None, negative_examples=None, pos_ids=None, neg_ids=None):
        file_name = f"{self.selected_theme}_{dict_hash(self.labels)}" #TODO Add user session ID

        
        # examples = list(self.positive_examples_collector.values())+list(self.negative_examples_collector.values())
        # ids = list(self.positive_examples_collector.keys())+list(self.negative_examples_collector.keys())
        # labels = [self.labels[x] for x in ids]

        examples = positive_examples+negative_examples
        ids = pos_ids+neg_ids
        labels = [1]*len(pos_ids) + [0]*len(neg_ids)

        df = patterns_against_examples(file_name=f"cache/{file_name}.csv",patterns=list(pattern_set.keys()), examples=examples, ids=ids, labels=labels, priority_phrases=self.negative_phrases, soft_match_on=self.soft_match_on, similarity_dict=self.similarity_dict, soft_threshold=self.soft_threshold)
        return df

    def ran_cache(self):
        file_name = f"{self.selected_theme}_{dict_hash(self.labels)}" 
        try:
            df = pd.read_csv(f"cache/{file_name}.csv")
            return df
        except:
            print("cache miss")
            return None
    ####### End Points ######

    def set_theme(self, theme):
        self.selected_theme = theme

        # self.data['positive'] = self.data['label'].apply(lambda x: x==self.selected_theme)
        # self.data['positive'] = self.data[theme].apply(lambda x: x==self.selected_theme)

        # self.data['positive'] = self.data[self.selected_theme]

        # self.clear_label()

        return self.get_labeled_dataset()
    
    def get_themes(self):
        return list(self.data.columns.unique())[2:]
    
    def get_selected_theme(self):
        return self.selected_theme


    def label_by_phrase(self, phrase, label):
        self.negative_phrases.append(nlp(phrase.strip()))
        # self.priority_unmatch.append(phrase)
        # print(list(self.negative_examples_collector.values())+self.negative_phrases)
        return {"status":200, "message":"ok", "phrase":phrase, "label":label}


    def add_theme(self, theme):
        self.themes[theme] = {}

        return list(self.themes.keys())


    def labeler(self, id, label):
        #check if label already exisits in the oposite collector and remove if it does
        exists = id in self.labels
        if(exists):
            previous_label = self.labels[id]
            if(previous_label==0):
                # remove from negative_example_collectore
                del self.negative_examples_collector[id]
            else:
                # remove from positive_example_collectore
                del self.positive_examples_collector[id]
        
        self.labels[id] = label
        sentence = nlp(self.data[self.data["id"] == id]["example"].values[0])
        if(label==0):
            self.negative_examples_collector[id] = sentence
        elif label==1:
            self.positive_examples_collector[id] = sentence
        
        return {"status":200, "message":"ok", "id":id, "label":label}

    def batch_label(self, id, label):
        #check if label already exisits in the oposite collector and remove if it does
        exists = id in self.labels
        if(exists):
            previous_label = self.labels[id]
            if(previous_label==0):
                # remove from negative_example_collectore
                del self.negative_examples_collector[id]
            else:
                # remove from positive_example_collectore
                del self.positive_examples_collector[id]
        
        self.labels[id] = label
        sentence = nlp(self.data[self.data["id"] == id]["example"].values[0])
        if(label==0):
            self.negative_examples_collector[id] = sentence
        elif label==1:
            self.positive_examples_collector[id] = sentence
        
        return {"status":200, "message":"ok", "id":id, "label":label}
    
    def clear_label(self):
        self.labels.clear()
        
        self.negative_examples_collector.clear()
        self.positive_examples_collector.clear()
        self.negative_phrases = []

        return {"message":"okay", "status":200}



    def get_labeled_dataset(self):
        dataset = []

        ids = self.data["id"].values
        for i in ids:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0].capitalize()
            item["true_label"] = self.data[self.data["id"] == i][self.selected_theme].values.tolist()[0] if self.selected_theme in self.data.columns else None
            item["score"] = None
            if(str(i) in self.labels):
                item["user_label"] = self.labels[str(i)]
            else:
                item["user_label"] = None
            # print()

            dataset.append(item)


        return dataset

    def all_patterns(self, depth=4, rewardThreshold=0.01, penalityThreshold=0.3):
        if len(self.labels.keys())==0 or len(self.positive_examples_collector.keys())==0:
            return {"message":"Nothing labeled yet"}

        #Check if data is in the chache
        cached = self.ran_cache()

        #For testing 
        cached = None

        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthh = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values())+self.negative_phrases, max_depth=depth, rewardThreshold=rewardThreshold, penalityThreshold=penalityThreshold, soft_match_on=self.soft_match_on, price=self.data, words_dict=self.words_dict, similarity_dict=self.similarity_dict,
            soft_threshold=self.soft_threshold)

            self.synthh.find_patters()

            

            try:
                df = self.save_cache(self.synthh.patterns_set)
            except:
                print(self.synthh.patterns_set)
                return {"message":"Annotate Some More"}
        
        patterns = get_patterns(df, self.labels)

        return patterns

    def resyntesize(self, depth=4, rewardThreshold=0.01, penalityThreshold=0.3):

        if len(self.labels.keys())==0 or len(self.positive_examples_collector.keys())==0:
            return {"message":"Nothing labeled yet"}

        #Check if data is in the chache    
        cached = self.ran_cache()
        
        #For testing 
        # cached = None
        if(type(cached) != type(None)): #for test
            df = cached
        else:
            self.synthh = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values())+self.negative_phrases, max_depth=depth, rewardThreshold=rewardThreshold, penalityThreshold=penalityThreshold, soft_match_on=self.soft_match_on, price=self.data, words_dict=self.words_dict, similarity_dict=self.similarity_dict,
            soft_threshold=self.soft_threshold)
            
            self.synthh.find_patters()

            try:
                df = self.save_cache(self.synthh.patterns_set)
            except:
               return {"message":"Annotate Some More"}
        
        res = train_linear_mode(df=df, data=self.data, theme=self.selected_theme, soft_match_on=self.soft_match_on,words_dict=self.words_dict, similarity_dict=self.similarity_dict, soft_threshold=self.soft_threshold)
        return res

    
    def get_related(self, id):

        score= self.synthesizer_collector[self.selected_theme].results['scores'][id] 
        explanation = self.synthesizer_collector[self.selected_theme].results['explanation']
        this_pattern_matches = []
        for key, value in explanation.items():
            if(value[id]!=""):
                this_pattern_matches.append(key)
        print("match related with ", this_pattern_matches)
        

        related = []

        related_highlights = {}

        for sentence_id in list(self.data['id'].values):
            if sentence_id == id:
                continue
            if self.synthesizer_collector[self.selected_theme].results['scores'][sentence_id] == score:
                related.append(sentence_id)
                related_highlights[sentence_id] = []
                for pattern in this_pattern_matches:
                    hglgt =  " ".join(explanation[pattern][sentence_id][0][0])
                    # print("related highlights ", explanation[pattern][sentence_id])
                    related_highlights[sentence_id].append(hglgt)
        # print("highlight ", related_highlights)





        
        dataset = []
        
        for i in related:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0]
            item["true_label"] = self.data[self.data["id"] == i][self.selected_theme].values.tolist()[0]
            item["score"] = None
            if(str(i) in self.labels):
                item["user_label"] = self.labels[str(i)]
            else:
                item["user_label"] = None
            # print()

            dataset.append(item)


        return dataset, related_highlights

    def run_test(self, iteration, no_annotation, depth=4, rewardThreshold=0.01, penalityThreshold=0.3):
        start_time = time.time()
        
        self.clear_label()
        pos_count = 0
        neg_count = 0
        collector = []
        newCollector = []
        self.clear_label()

        all_ids  = self.data["id"].values.tolist()

        ids = random.sample(all_ids, no_annotation)
        annotation = []
        for id in ids:
            annotation.append(self.data[self.data["id"]==id]["positive"].values[0])
        
            for i, lbl in zip(ids, annotation):
                self.labeler(str(i), int(lbl))

                if lbl ==1:
                    pos_count+=1
                elif lbl==0:
                    neg_count+=1
        print(self.labels)
        self.words_dict, self.similarity_dict = get_similarity_dict(self.data["example"].values, soft_threshold=self.soft_threshold)
        print("words dict finished")
        for x in range(iteration):
            iteration_start_time = time.time()
            print("Starting Synthesizing")
            results = self.resyntesize(depth=depth, rewardThreshold=rewardThreshold, penalityThreshold=penalityThreshold)

            print("Finishing Synthesizing")
            
                


            ids = [str(x) for x in random.sample(all_ids, no_annotation)]

            annotation = []
            for id in ids:
                # all_ids.remove(id)
                annotation.append(self.data[self.data["id"]==id]["positive"].values[0])


            # temp = dict()
            # temp["fscore"] = results["fscore"]
            # temp["recall"] = results["recall"]
            # temp["precision"] = results["precision"]

            # temp["overall_fscore"] = results["overall_fscore"]
            # temp["overall_recall"] = results["overall_recall"]
            # temp["overall_precision"] = results["overall_precision"]

            results["positive_annotated"] = len(list(self.positive_examples_collector.keys()))
            results["negative_annotated"] = len(list(self.negative_examples_collector.keys()))

            
            results["positive_annotated_examples"] = [str(x) for x in self.positive_examples_collector.values()]

            results["negative_annotated_examples"] = [str(x) for x in self.negative_examples_collector.values()]
            
            newresults = {}
            newresults["iteration_num"] = x
            newresults["iteration_time"] = '{:.1f} minutes'.format((time.time() - iteration_start_time) / 60)
            print(newresults["iteration_time"])
            # if 'scores' in results:
            #     dataset = self.data
            #     baseline = pd.read_json("results/baseline_result_15_5.json")
            #     resultA = [v > 0.5 for k,v in results['scores'].items()]
            #     resultB = [v > 0.5 for k,v in baseline['scores'][x].items()]
            #     newresults['baseline_iteration_time'] = baseline["iteration_time"][x]
            #     newresults['Fscore_comparison'] = {'This': results['overall_fscore'], 'Baseline': baseline['overall_fscore'][x]}
            #     newresults["TP"] = []
            #     newresults["FP"] = []
            #     newresults["TN"] = []
            #     newresults["FN"] = []
            #     for j in range(len(resultA)):
            #         if resultA[j] != resultB[j]:
            #             ex = [dataset['example'][j]]
            #             for pat in results['patterns']:
            #                 if results['explanation'][pat['pattern']][dataset['id'][j]] != '':
            #                     ex.append({pat['pattern']:results['explanation'][pat['pattern']][dataset['id'][j]], "weight":pat['weight']})
            #             # ex_baseline = [dataset['example'][j]]
            #             # for pat in baseline['patterns'][x]:
            #             #     if baseline['explanation'][x][pat['pattern']][dataset['id'][j]] != '':
            #             #         ex_baseline.append({pat['pattern']:baseline['explanation'][x][pat['pattern']][dataset['id'][j]]})
            #             #         print(ex_baseline)
            #             if dataset['positive'][j] == resultA[j]:
            #                 if resultA[j] == 1:
            #                     newresults["TP"].append({dataset['id'][j]:ex})
            #                 else: newresults["TN"].append({dataset['id'][j]:ex})
            #             if dataset['positive'][j] != resultA[j]:
            #                 if resultA[j] == 1:
            #                     newresults["FP"].append({dataset['id'][j]:ex})
            #                 else: newresults["FN"].append({dataset['id'][j]:ex})
            #     newresults['TP_num'] = len(newresults['TP'])
            #     newresults['FP_num'] = len(newresults['FP'])
            #     newresults['TN_num'] = len(newresults['TN'])
            #     newresults['FN_num'] = len(newresults['FN'])
            results = {**results, **newresults}
            # Overview
            # results = newresults

            collector.append(results)
            newCollector.append(newresults)

            for i, lbl in zip(ids, annotation):
                self.labeler(str(i), int(lbl))

                if lbl ==1:
                    pos_count+=1
                elif lbl==0:
                    neg_count+=1
        

        

        # collector += newCollector
        with open('results/test_results.json', 'w') as f:
            json.dump(collector, f)
        print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))

        return collector

    def explain_pattern(self, pattern):
        exp = {}
        pattern = pattern.replace('+', ', ')
        pattern = pattern.replace('|', ', ')
        pattern_list = pattern.split(", ")
        for pat in pattern_list:
            pattern_expanded = expand_working_list(pat, soft_match_on=True, similarity_dict=self.similarity_dict)[0][0]
            if('LEMMA' in pattern_expanded and pat[0]=='('):
                print(pattern_expanded['LEMMA'])
                exp[pat] = pattern_expanded['LEMMA']["IN"] 
        return exp

        


    def testing_cache_new(self):
        pos_count = 0
        neg_count = 0
        collector = []
        self.clear_label()

        ids = random.sample(range(0, 600), 50)
        annotation = []
        for id in ids:
            annotation.append(self.data[self.data["id"]==id]["positive"].values[0])
        for i, lbl in zip(ids, annotation):

            self.labeler(str(i), int(lbl))
            if lbl ==1:
                pos_count+=1
            elif lbl==0:
                neg_count+=1
            print(self.labels)
            
            results = self.resyntesize()
            
            temp = dict()
            temp["fscore"] = results["fscore"]
            temp["recall"] = results["recall"]
            temp["precision"] = results["precision"]

            temp["overall_fscore"] = results["overall_fscore"]
            temp["overall_recall"] = results["overall_recall"]
            temp["overall_precision"] = results["overall_precision"]

            temp["positive_annotated"] = pos_count
            temp["negative_annotated"] = neg_count
            collector.append(temp)
        
        with open('results_20_binary_take2.json', 'w') as f:
            json.dump(collector, f)


        return collector



######################################################################################################################################################

    def initialize_synthesizers(self, themes):
        for theme in themes:
            self.synthesizer_collector[theme] = Synthesizer(positive_examples = [], negative_examples = [], soft_match_on=self.soft_match_on, words_dict=self.words_dict, similarity_dict=self.similarity_dict,
            soft_threshold=self.soft_threshold)
    
    def label_element(self, elementId, label):
        if elementId in self.element_to_label:
            self.element_to_label[elementId].append(label)
        else:
            self.element_to_label[elementId] = [label]
            
        if label in self.theme_to_element:
            self.theme_to_element[label].append(elementId)
        else:
            self.theme_to_element[label] = [elementId]
        
        if elementId not in self.element_to_sentence:
            sentence = nlp(self.data[self.data["id"] == elementId]["example"].values[0])
            self.element_to_sentence[elementId] = sentence

        # print(self.theme_to_element)
        # print(self.element_to_label)
        return {"status":200, "message":"ok", "id":elementId, "label":label}
    
    def delete_label(self, elementId, label):
        self.element_to_label[elementId].remove(label)

        self.theme_to_element[label].remove(elementId)

        print(self.theme_to_element)
        print(self.element_to_label)
        return {"status":200, "message":"label deleted", "id":elementId, "label":label}

    def synthesize_patterns(self):
        #aggregate examples
        try:
            positive_examples_id = self.theme_to_element[self.selected_theme]
        except:
            response = {}
            response["message"] = f"Nothing labeled for {self.selected_theme}"
            response["status_code"] = 404
            return response
        positive_examples = []
        for id in positive_examples_id:
            positive_examples.append(self.element_to_sentence[id])
        
        negative_examples_id = []
        negative_examples = []
        for elementId in self.element_to_label:
            if(not self.selected_theme in self.element_to_label[elementId]):
                negative_examples.append(self.element_to_sentence[elementId])
                negative_examples_id.append(elementId)

    
        #sanity check
        print("POS ", positive_examples)
        print("NEG ", negative_examples)
        

        #check if we have data annotated
        if len(positive_examples)==0:
            response = {}
            response["message"] = "Annotate Some More"
            response["status_code"] = 404
            return response
        
        #Check if data is in the chache
        cached = self.ran_cache()

        #For testing we will set the cache to none
        cached = None

        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthesizer_collector[self.selected_theme].set_params(positive_examples, negative_examples)
            self.synthesizer_collector[self.selected_theme].find_patters()

            

            # df = self.save_cache(self.synthesizer_collector[self.selected_theme].patterns_set, positive_examples, negative_examples, positive_examples_id, negative_examples_id)
            try:
                df = self.save_cache(self.synthesizer_collector[self.selected_theme].patterns_set, positive_examples, negative_examples, positive_examples_id, negative_examples_id)
            except:
                response = {}
                response["message"] = "Annotate Some More"
                response["status_code"] = 404

                # print(self.synthesizer_collector[self.selected_theme].patterns_set)
                return response
        
        patterns = get_patterns(df, self.labels)

        return patterns
    
    def get_linear_model_results(self):
        #check if patterns are in cache, this will most likely be true because in most cases the above 
        #function will be called before this function
        #if patterns are not cached we will synthesize and cache them in this function

        #Check if data is in the chache    
        cached = self.ran_cache()

        #For testing 
        cached = None
        if(type(cached) != type(None)): #for test
            df = cached
        else:
            res = self.synthesize_patterns()
            
            if("status_code" in res and res["status_code"]==404):
                return res
            
            cached = self.ran_cache() 
            #TODO annotate some more data needs to be added here and maybe a way to check if we have enough data annotated
            df = cached
            if(type(df) == type(None)):
                #TODO handle this in a better way, but if the code gets here we know that no patterns have been synthesized becuase there weren't enough annotations
                return {"message": f"Not enough annotations for {self.selected_theme}"}
        
        res = train_linear_mode(df=df, data=self.data, theme=self.selected_theme, soft_match_on=self.soft_match_on,words_dict=self.words_dict, similarity_dict=self.similarity_dict, soft_threshold=self.soft_threshold)
        return res
    
    # def get_cache_name():



        #check if enough examples have been annotated
        # if len(self.labels.keys())==0 or len(self.positive_examples_collector.keys())==0:
        #     return {"message":"Nothing labeled yet"}

    def run_multi_label_test(self, iteration, no_annotation):
        #I declare thee a results collector
        collector = []

        #keep track of how many are being annotated
        total_annotation_count = 0
        theme_annotation_count = {}
        all_themes = self.get_themes()

        for theme in all_themes:
            theme_annotation_count[theme] = 0


        #get all ids
        all_ids  = self.data["id"].values.tolist()


        for i in range(iteration):
            #pick random ids to annotate
            ids = random.sample(all_ids, no_annotation)

            #for each id picked annotate all the positive labels
            for i in range(len(ids)):
                total_annotation_count += 1
                for theme in all_themes:
                    if(self.data[self.data['id']== ids[i]][theme].values[0]):
                        self.label_element(ids[i], theme)
                        theme_annotation_count[theme] += 1 
                        print(f"labeled example {ids[i]} as {theme}")

        #synthesize and come up with scores for each theme
        all_themes_results = []
        for theme in all_themes:
            print(f"working with {theme}")
            self.set_theme(theme)
            temp = {}
            try:
                results = self.get_linear_model_results()
            except Exception as Argument:
                # creating/opening a file
                f = open("errorlog.txt", "a")
            
                # writing in the file
                f.write(str(Argument))
                
                # closing the file
                f.close()  

            
            #collect only relevant information from the results
            try:
                temp["theme"] = theme
                temp["fscore"] = results["fscore"]
                temp["recall"] = results["recall"]
                temp["precision"] = results["precision"]
                temp["overall_fscore"] = results["overall_fscore"]
                temp["overall_recall"] = results["overall_recall"]
                temp["overall_precision"] = results["overall_precision"]
                temp["patterns"] = results["patterns"]
                temp["weights"] = results["weights"]
                temp["total_annotation_count"] = total_annotation_count
                temp["annotation_per_theme"] = theme_annotation_count
            except:
                temp["theme"] = theme
                temp["message"] = results

            all_themes_results.append(temp)
        
        collector.append(all_themes_results)


        #aggregate scores

        #return resuts
        return collector

