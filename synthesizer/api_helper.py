from synthesizer.linear_network import patterns_against_examples, train_linear_mode
from synthesizer.penality_based_threaded import Synthesizer
from synthesizer.helpers import dict_hash
from synthesizer.helpers import get_patterns
import pandas as pd
import json
import spacy
import random

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

        # self.theme = "price_service"
        # self.theme = "hate_speech_binary"

        self.data = pd.read_csv(f"examples/df/{self.theme}.csv")
        self.data['positive'] = self.data['label'].apply(lambda x: x==self.selected_theme)
        self.priority_unmatch = []
        self.priority_match = []
        self.labels = {}
        self.themes = {}

        self.results = {}
    
    
    def save_cache(self, pattern_set):
        file_name = dict_hash(self.labels)
        examples = list(self.positive_examples_collector.values())+list(self.negative_examples_collector.values())
        ids = list(self.positive_examples_collector.keys())+list(self.negative_examples_collector.keys())
        labels = [self.labels[x] for x in ids]

        df = patterns_against_examples(file_name=f"cache/{file_name}.csv",patterns=list(pattern_set.keys()), examples=examples, ids=ids, labels=labels, priority_phrases=self.negative_phrases)
        return df

    def ran_cache(self):
        file_name = dict_hash(self.labels)
        try:
            df = pd.read_csv(f"cache/{file_name}.csv")
            return df
        except:
            print("cache miss")
            return None
    ####### End Points ######

    def set_theme(self, theme):
        self.selected_theme = theme

        self.data['positive'] = self.data['label'].apply(lambda x: x==self.selected_theme)
        self.clear_label()
        
        return self.get_labeled_dataset()

    def get_themes(self):
        return list(self.data['label'].unique())
    
    def get_selected_theme(self):
        return self.selected_theme


    def label_by_phrase(self, phrase, label):
        self.negative_phrases.append(nlp(phrase.strip()))
        # self.priority_unmatch.append(phrase)
        # print(list(self.negative_examples_collector.values())+self.negative_phrases)
        return {"status":200, "message":"ok", "phrase":phrase, "label":label}

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
            item["example"] = self.data[self.data["id"] == i]["example"].values[0]
            item["true_label"] = self.data[self.data["id"] == i]["positive"].values.tolist()[0]
            item["score"] = None
            if(str(i) in self.labels):
                item["user_label"] = self.labels[str(i)]
            else:
                item["user_label"] = None
            # print()

            dataset.append(item)


        return dataset

    def all_patterns(self):
        if len(self.labels.keys())==0 or len(self.positive_examples_collector.keys())==0:
            return {"message":"Nothing labeled yet"}

        #Check if data is in the chache
        cached = self.ran_cache()

        #For testing 
        # cached = None

        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthh = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values())+self.negative_phrases, max_depth=3)

            self.synthh.find_patters()
            try:
                df = self.save_cache(self.synthh.patterns_set)
            except:
                return {"message":"Annotate Some More"}
        
        patterns = get_patterns(df, self.labels)

        return patterns

    def resyntesize(self, depth=4, rewardThreshold=0.01, penalityThreshold=0.3):

        if len(self.labels.keys())==0 or len(self.positive_examples_collector.keys())==0:
            return {"message":"Nothing labeled yet"}

        #Check if data is in the chache    
        cached =  self.ran_cache()

        #For testing 
        # cached = None
        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthh = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values())+self.negative_phrases, max_depth=depth, rewardThreshold=rewardThreshold, penalityThreshold=penalityThreshold)
            
            self.synthh.find_patters()
            try:
                df = self.save_cache(self.synthh.patterns_set)
            except:
               return {"message":"Annotate Some More"} 
        

        res = train_linear_mode(df=df, price=self.data)
        self.results = res
        
        return res

    def get_related(self, id):

        # print(self.results)

        
        # explanation = self.results['explanation']
        # sentence_explanation = []

        
        # for key,value in explanation.items():
        #     sentence_explanation.append({key:value[id]})

        score = self.results['scores'][id]

        related = []

        for sentence_id in list(self.data['id'].values):
            if sentence_id == id:
                continue
            if self.results['scores'][sentence_id] == score:
                related.append(sentence_id)
        
        dataset = []
        
        for i in related:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0]
            item["true_label"] = self.data[self.data["id"] == i]["positive"].values.tolist()[0]
            item["score"] = None
            if(str(i) in self.labels):
                item["user_label"] = self.labels[str(i)]
            else:
                item["user_label"] = None
            # print()

            dataset.append(item)

        # related = [df["id"]==x f]

        
        print(related)


        return dataset

    def run_test(self, iteration, no_annotation, depth=4, rewardThreshold=0.01, penalityThreshold=0.3):
        self.clear_label()
        pos_count = 0
        neg_count = 0
        collector = []
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
        for x in range(iteration):
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


            collector.append(results)


            for i, lbl in zip(ids, annotation):
                self.labeler(str(i), int(lbl))

                if lbl ==1:
                    pos_count+=1
                elif lbl==0:
                    neg_count+=1
        
        with open('results/test_results.json', 'w') as f:
            json.dump(collector, f)


        return collector



