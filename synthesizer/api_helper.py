from synthesizer.linear_network import patterns_against_examples, train_linear_mode
from synthesizer.penality_based_threaded import Synthesizer
from synthesizer.helpers import dict_hash
from synthesizer.helpers import get_patterns
import pandas as pd
import json
class APIHelper:
    def __init__(self):
        self.positive_examples = []
        self.negative_examples = []
        self.theme = "price"
        self.synthh = None # Synthesizer(positive_examples = self.positive_examples, negative_examples = self.negative_examples)
        self.labeled = 0


        self.data = pd.read_csv(f"examples/df/{self.theme}.csv")
        self.examples = [] #["This particular location has a good check in deal.","some of the items they sale here are a bit over priced but if you don't mind paying a bit extra this is the place to go."]
        self.positive_examples = [] #["This particular location has a good check in deal.","some of the items they sale here are a bit over priced but if you don't mind paying a bit extra this is the place to go."]
        self.labels = {} #{"1":1, "2":1}

        self.annotation = {"1":1, "2":1, "3":0, "4":0, "5":0,"6":1, "7":1, "8":0, "9":0, "10":1}

    def labeler(self, id, label):
        self.labeled+=1
        # if(id in self.annotation.keys()): #For testin
        #     label = self.annotation[id]
        #     if(id=="3"):
        #         label = 1
        #     elif(id =="4"):
        #         self.labels["3"] = 0

        if(id in self.labels.keys()):
            self.labels[id] = label
            return {"message":"okay", "status":200}
        
        self.labels[id] = label
        self.examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])

        print(self.labels)
        return {"message":"okay", "status":200}
    
    def clear_label(self):
        self.labels.clear()
        self.labeled = 0
        self.examples =[]

        return {"message":"okay", "status":200}
    
    def check_cache(self):
        file_name = dict_hash(self.labels)
        try:
            with open(f"cache/{file_name}", "r") as file:
                patterns = json.load(file)
                return patterns
                
        except:
            print("cache miss")
            return None
    
    def ran_cache(self):

        file_name = dict_hash(self.labels)
        try:
            df = pd.read_csv(f"cache/{file_name}.csv")
            return df
        except:
            print("cache miss")
            return None

    def save_cache(self, pattern_set, examples):
        file_name = dict_hash(self.labels)
        df = patterns_against_examples(file_name=f"cache/{file_name}.csv",patterns=list(pattern_set.keys()), examples=examples, ids=self.labels.keys(), labels=self.labels.values())
        return df
        
        # with open(f"cache/{file_name}", "w") as file:
        #     json.dump(pattern_set, file)
        
    #returns "Pattern":[precision, recall, fscore]
    def resyntesize(self):
        cached = self.ran_cache()
        if(type(cached) != type(None)):
            df = cached
        else:
            self.positive_examples = []
            self.negative_examples = []
            
            for id in self.labels.keys():
                if(self.labels[id]==1):
                    self.positive_examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])
                else:
                    self.negative_examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])
            
                        
            self.synthh = Synthesizer(positive_examples = self.positive_examples, negative_examples = self.negative_examples)
            self.synthh.find_patters()
            df = self.save_cache(self.synthh.patterns_set, examples=self.examples)
        

        res = train_linear_mode(df=df, price=self.data)
        return res

    def testing_cache(self):
        pos_count = 0
        neg_count = 0
        collector = []
        annotation = {"1":1, "2":1, "3":0, "4":0, "5":0,"6":1, "7":1, "8":0, "9":0, "10":1 ,"11":1,"12":1, "13":1, "14":1, "15":0, "16":0, "17":0, "18":0, "19":1, "20":1, "22":1, "23":1, "24":0, "25": 0 }
        self.clear_label()
        for i in annotation.keys():
            self.labeler(i, annotation[i])
            if annotation[i] ==1:
                pos_count+=1
            elif annotation[i]==0:
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
    
    def all_patterns(self):
        # self.clear_label()
        # self.labeler("1", 1)
        if len(self.labels.keys())==0:
            return {"message":"Nothing labeled yet"}
        cached = self.ran_cache()
        if(type(cached) != type(None)):
            df = cached
        else:
            self.positive_examples = []
            self.negative_examples = []
            
            for id in self.labels.keys():
                if(self.labels[id]==1):
                    self.positive_examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])
                else:
                    self.negative_examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])
            
                        
            self.synthh = Synthesizer(positive_examples = self.positive_examples, negative_examples = self.negative_examples)
            self.synthh.find_patters()
            df = self.save_cache(self.synthh.patterns_set, examples=self.examples)
        
        patterns = get_patterns(df, self.labels)

        return patterns
    
    def get_dataset(self):
        dataset = dict()
        ids = self.data["id"].values

        for i in ids:
            dataset[str(i)] = self.data[self.data["id"] == i]["example"].values[0]


        return dataset
    
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

