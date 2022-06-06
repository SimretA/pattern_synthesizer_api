from __future__ import annotations
from synthesizer_v2.linear_network import patterns_against_examples, train_linear_mode
from synthesizer_v2.penality_based_threaded import Synthesizer
from synthesizer_v2.helpers import dict_hash
from synthesizer_v2.helpers import get_patterns
import pandas as pd
import json
import spacy
import numpy as np
import random

nlp = spacy.load("en_core_web_sm")

class ThemeSynthesizer:
    def __init__(self, theme_name, positive_examples_collector, negative_examples_collector, labels={}):
        self.theme_name = theme_name
        self.positive_examples_collector = positive_examples_collector
        self.negative_examples_collector = negative_examples_collector
        self.all_patterns = None
        self.labels = {}
        self.sytnthesizer = None
        self.patterns = None
        self.linear_model = None
        self.meta = None
    def set_positive_example(self, positive_examples_collector):
        self.positive_examples_collector = positive_examples_collector

    def set_negative_example(self, negative_examples_collector):
        self.negative_examples_collector


    def save_cache(self, pattern_set):
        file_name = dict_hash(self.labels)
        print("file name is ", file_name)
        examples = list(self.positive_examples_collector.values())+list(self.negative_examples_collector.values())
        ids = list(self.positive_examples_collector.keys())+list(self.negative_examples_collector.keys())
        # labels = [self.labels[x] for x in ids]
        labels = [1]*len(self.positive_examples_collector.values()) + [0]*len(self.negative_examples_collector.values())

        df = patterns_against_examples(file_name=f"cache/{self.theme_name}_{file_name}.csv",patterns=list(pattern_set.keys()), examples=examples, ids=ids, labels=labels)
        return df
    
    def resynthesize(self, data):
        file_name = dict_hash(self.labels)
        try:
            df = pd.read_csv(f"cache/{self.theme_name}_{file_name}.csv")
        except:
                
            data["positive"] = data["label"].apply(lambda x: 1 if x==self.theme_name else 0)
            print("DEBUGING, ", self.theme_name, data["positive"].values )
            self.sytnthesizer = Synthesizer(positive_examples = list(self.positive_examples_collector.values()), negative_examples = list(self.negative_examples_collector.values()))
            self.sytnthesizer.find_patters()
            self.patterns = self.sytnthesizer.patterns_set
            df = self.save_cache(self.sytnthesizer.patterns_set)
        all_patterns = get_patterns(df, df["labels"].values)
        # print("All patterns ",all_patterns)

        result = train_linear_mode(df=df, price=data)
        self.meta = res = result[1]
        self.linear_model = result[0] 
        res["all_patterns"] = all_patterns

        return res

class APIHelper2:
    def __init__(self):
        self.positive_examples_collector = {}
        self.negative_examples_collector = {}
        # self.theme = "hatexplain"
        # self.theme = "hate_speech"
        self.theme = "price_service"
        # self.theme = "hatexplain_small"
        self.data = pd.read_csv(f"examples/df/{self.theme}.csv")
        self.labels = {}
        self.theme_to_id = {}
        self.id_to_theme = {}
        self.themeid_to_examples_collector = {}
    
    
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
        id = len(self.theme_to_id)
        self.theme_to_id[theme] = id
        self.id_to_theme[id] = theme
        self.themeid_to_examples_collector[id] = {}

        return self.theme_to_id


    def labeler(self, id, label):
        #check if label exists in themes
        if(label not in self.theme_to_id):
            self.add_theme(label)
            print("Theme added")
            # return {"Error": "Theme does not exist"}
        
        theme_id = self.theme_to_id[label]

        #check if label already exisits in the collector
        exists = id in self.labels
        if(exists):
            # previous_label = self.labels[id]
            # del self.themeid_to_examples_collector[previous_label][id]
            self.labels[id].append(theme_id)
        else:
             self.labels[id] = [theme_id]
        sentence = nlp(self.data[self.data["id"] == id]["example"].values[0])
        

        self.themeid_to_examples_collector[theme_id][id] = sentence
        
        
        return {"status":200, "message":"ok"}
    
    def remove_label(self, id, label):

        return {"status":200, "message":"ok"}
    
    def clear_label(self):
        self.labels.clear()
        self.themeid_to_examples_collector.clear()
        self.theme_to_id.clear()
        self.id_to_theme.clear()

        return {"message":"okay", "status":200}



    def get_labeled_dataset(self):
        dataset = []
        print(self.labels)

        ids = self.data["id"].values
        for i in ids:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0]
            item["true_label"] = self.data[self.data["id"] == i]["label"].values.tolist()[0]
            if(str(i) in self.labels):
                item["user_label"] = [self.id_to_theme[x] for x in self.labels[str(i)]]
            else:
                item["user_label"] = None
            # print()

            dataset.append(item)


        return dataset

    def all_patterns(self):
        if len(self.labels.keys())==0 or len(self.themeid_to_examples_collector.keys())==0:
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

        collection = {}
        results = {}
        print(self.themeid_to_examples_collector)

        for i in self.theme_to_id:
            id = self.theme_to_id[i]

            positives = self.themeid_to_examples_collector[id]

            negatives = {}
            for x in self.themeid_to_examples_collector:
                if(x!=id):
                    negatives = {**negatives, **self.themeid_to_examples_collector[x]}

            if(len(positives)==0):
                continue

            

            collection[i] = ThemeSynthesizer(i, 
            positive_examples_collector=positives, 
            negative_examples_collector=negatives,
            labels=self.themeid_to_examples_collector)

            res = collection[i].resynthesize(self.data)

            results[i] = res

        return results
    
    def get_next_data(self):
        self.clear_label()
        if(self.theme=="hate_speech"):
            self.theme ="price_service"
        elif(self.theme=="price_service"):
            self.theme = "hate_speech"
        self.data = pd.read_csv(f"examples/df/{self.theme}.csv")
        return self.get_labeled_dataset()
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

        self.clear_label()

        self.add_theme("offensive")
        self.add_theme("hate")
        self.add_theme("none")


        price_count = 0
        service_count = 0
        environment_count = 0
        collector = []
        
        ids = [str(x) for x in random.sample(range(0, 600), 60)]
        annotation = []
        for id in ids:
            annotation.append(self.data[self.data["id"]==id]["label"].values[0])
        # annotation = {"0":"price", "1":"price"}#, "2":"price", "3":"service", "4":"service", "5":"price", "7":"price", "14":"environment", "22":"environment", "16":"service", "15":"service",  "23":"service", "30":"price", "31":"price",  "33":"price",  "34":"price", "37":"service", "38":"service", "39":"environment" }
        
        for i, lbl in zip(ids, annotation):

            self.labeler(i, lbl)
            if lbl =="offensive":
                price_count+=1
            elif lbl=="hate":
                service_count+=1
            elif lbl=="none":
                environment_count+=1
            
            print(self.themeid_to_examples_collector)
            

            temp = self.resyntesize()
            # temp = dict()
            # temp["fscore"] = results["fscore"]
            # temp["recall"] = results["recall"]
            # temp["precision"] = results["precision"]

            # temp["overall_fscore"] = results["overall_fscore"]
            # temp["overall_recall"] = results["overall_recall"]
            # temp["overall_precision"] = results["overall_precision"]

            temp["offensive_count"] = price_count
            temp["hate_count"] = service_count
            temp["none_count"] = environment_count
            collector.append(temp)
        with open('results_40_may18.json', 'w') as f:
            json.dump(collector, f)


        return collector
    
    def testing_label_ordering(self):

        self.clear_label()

        # self.add_theme("price")
        # self.add_theme("service")
        # self.add_theme("environment")

        self.add_theme("offensive")
        self.add_theme("hatespeech")
        self.add_theme("normal")
        self.add_theme("none")


        price_count = 0
        service_count = 0
        environment_count = 0
        collector = []
        all_ids  = self.data["id"].values.tolist()
        
        ids = [str(x) for x in random.sample(all_ids, 10)] #Get 5 annotations randomly first
        annotation = []
        for id in ids:
            # all_ids.remove(id)

            annotation.append(self.data[self.data["id"]==id]["label"].values[0])
        
        print("Starting Annotation")
        for i, lbl in zip(ids, annotation):

            self.labeler(i, lbl)
            # if lbl =="price":
            #     price_count+=1
            # elif lbl=="service":
            #     service_count+=1
            # elif lbl=="environment":
            #     environment_count+=1
            
            if lbl =="offensive":
                price_count+=1
            elif lbl=="hatespeech":
                service_count+=1
            elif lbl=="normal":
                environment_count+=1

            # print(self.themeid_to_examples_collector)
        print("Finishing Annotation")
        
        for x in range(10):
            print("Starting synthesizing")
            temp = self.resyntesize() #Synthesize and annotate next batch of 5 based on the prediction
            print("Finishing synthesizing")

            next_batch = []

            # k= 5
            # for i in temp:
            #     arr = np.asarray(temp[i]["scores"])
                
            #     idd = np.argpartition(arr, len(arr) - k)[-k:]
                # next_batch += idd.tolist()
            next_batch = [str(x) for x in random.sample(all_ids, 10)]
            # next_batch = []

            annotation = []
            for id in next_batch:
                # all_ids.remove(id)
                annotation.append(self.data[self.data["id"]==id]["label"].values[0])

            ############ 
            # 

            # temp["price_count"] = price_count
            # temp["service_count"] = service_count
            # temp["environment_count"] = environment_count  


            temp["offensive_count"] = price_count
            temp["hate_count"] = service_count
            temp["none_count"] = environment_count
            examples = {self.id_to_theme[k]:str(v) for k,v in self.themeid_to_examples_collector.items() }
            
            temp["annotated"] = examples

            collector.append(temp)

            # print(self.themeid_to_examples_collector)
            # print(collector)

            ############

            for i, lbl in zip(next_batch, annotation):

                self.labeler(i, lbl)

                # if lbl =="price":
                #     price_count+=1
                # elif lbl=="service":
                #     service_count+=1
                # elif lbl=="environment":
                #     environment_count+=1


                if lbl =="offensive":
                    price_count+=1
                elif lbl=="hatespeech":
                    service_count+=1
                elif lbl=="normal":
                    environment_count+=1
                
        with open('results_collector/hatexplan_random_selection_take2.json', 'w') as f:
            json.dump(collector, f)

 
        return collector


