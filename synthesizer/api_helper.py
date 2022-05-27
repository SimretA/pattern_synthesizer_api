from synthesizer.linear_network import patterns_against_examples, train_linear_mode
from synthesizer.penality_based_threaded import Synthesizer
from synthesizer.helpers import dict_hash
from synthesizer.helpers import get_patterns
import pandas as pd
import json
import spacy
import random

nlp = spacy.load("en_core_web_sm")
        

class APIHelper:
    def __init__(self):
        self.positive_examples_collector = {}
        self.negative_examples_collector = {}
        self.theme = "price_service"

        # self.theme = "hate_speech_binary"

        self.data = pd.read_csv(f"examples/df/{self.theme}.csv")
        self.labels = {}
        self.themes = {}
    
    
    def save_cache(self, pattern_set):
        file_name = dict_hash(self.labels)
        examples = list(self.positive_examples_collector.values())+list(self.negative_examples_collector.values())
        ids = list(self.positive_examples_collector.keys())+list(self.negative_examples_collector.keys())
        labels = [self.labels[x] for x in ids]

        df = patterns_against_examples(file_name=f"cache/{file_name}.csv",patterns=list(pattern_set.keys()), examples=examples, ids=ids, labels=labels)
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
        sentence = nlp(self.data[self.data["id"] == int(id)]["example"].values[0])
        if(label==0):
            self.negative_examples_collector[id] = sentence
        elif label==1:
            self.positive_examples_collector[id] = sentence
        
        return {"status":200, "message":"ok"}
    
    def clear_label(self):
        self.labels.clear()
        
        self.negative_examples_collector.clear()
        self.positive_examples_collector.clear()

        return {"message":"okay", "status":200}



    def get_labeled_dataset(self):
        dataset = []

        ids = self.data["id"].values
        for i in ids:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0]
            item["true_label"] = self.data[self.data["id"] == i]["positive"].values.tolist()[0]
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
        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthh = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values()))

            self.synthh.find_patters()
            

            df = self.save_cache(self.synthh.patterns_set)
        
        patterns = get_patterns(df, self.labels)

        return patterns

    def resyntesize(self):

        if len(self.labels.keys())==0 or len(self.positive_examples_collector.keys())==0:
            return {"message":"Nothing labeled yet"}

        #Check if data is in the chache    
        cached = self.ran_cache()
        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthh = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values()))
            
            self.synthh.find_patters()
            df = self.save_cache(self.synthh.patterns_set)
        

        res = train_linear_mode(df=df, price=self.data)
        return res

    def test(self):
        pos_count = 0
        neg_count = 0
        collector = []
        annotation = {"1":1, "2":1, "3":0, "4":0, "5":0}#,"6":1, "7":1, "8":0, "9":0, "10":1 ,"11":1,"12":1, "13":1, "14":1, "15":0, "16":0, "17":0, "18":0, "19":1, "20":1, "22":1, "23":1, "24":0, "25": 0 }
        self.clear_label()
        for i in annotation.keys():
            self.labeler(i, annotation[i])
            if annotation[i] ==1:
                pos_count+=1
            elif annotation[i]==0:
                neg_count+=1
            print(self.labels)
        results = self.resyntesize()
        return results
    def testing_cache(self):
        pos_count = 0
        neg_count = 0
        collector = []
        annotation = {"1":1, "2":1, "3":0, "4":0, "5":0,"6":1, "7":1, "8":0, "9":0, "10":1 ,"11":1,"12":1, "13":1, "14":1, "15":0, "16":0, "17":0, "18":0, "19":1, "20":1, "22":1, "23":1, "24":0, "25": 0 }
        self.clear_label()
        for i in annotation.keys():

            lbl = self.data[self.data["id"]==int(i)]["positive"].tolist()[0]
            self.labeler(i, lbl)
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


        return collector



    def testing_cache_new(self):
        pos_count = 0
        neg_count = 0
        collector = []
        # annotation = {"1":1, "2":1, "3":0, "4":0, "5":0,"6":1, "7":1, "8":0, "9":0, "10":1 ,"11":1,"12":1, "13":1, "14":1, "15":0, "16":0, "17":0, "18":0, "19":1, "20":1, "22":1, "23":1, "24":0, "25": 0 }
        self.clear_label()

        ids = random.sample(range(0, 600), 50)
        annotation = []
        for id in ids:
            annotation.append(self.data[self.data["id"]==id]["positive"].values[0])
        for i, lbl in zip(ids, annotation):

            # lbl = self.data[self.data["id"]==int(i)]["positive"].tolist()[0]
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


