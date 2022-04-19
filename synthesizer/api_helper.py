from synthesizer.linear_network import patterns_against_examples, train_linear_mode
from synthesizer.penality_based_threaded import Synthesizer
from synthesizer.helpers import dict_hash
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
        self.examples = ["This particular location has a good check in deal.","some of the items they sale here are a bit over priced but if you don't mind paying a bit extra this is the place to go."]
        self.positive_examples = ["This particular location has a good check in deal.","some of the items they sale here are a bit over priced but if you don't mind paying a bit extra this is the place to go."]
        self.labels = {"1":1, "2":1}


    def labeler(self, id, label):
        self.labeled+=1
        self.labels[id] = label
        self.examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])
        # if(id=="30"): #TODO remove. here for testing
        #     for i in range(1,73):
        #         id = str(i)
        #         self.labeled+=1
        #         self.labels[id] = 1 if i<=39 else 0
        #         self.examples.append(self.data[self.data["id"] == int(id)]["example"].values[0])


        # print(len(self.examples))
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

        res = train_linear_mode(df=df)
        return res

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


