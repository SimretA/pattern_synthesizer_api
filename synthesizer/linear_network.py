import pandas as pd
import numpy as np
import torch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")

from synthesizer.helpers import expand_working_list
import json

def check_matching(sent, working_list):
    matcher = Matcher(nlp.vocab)
    for index, patterns in enumerate(working_list):
        matcher.add(f"rule{index}", [patterns])
    doc = nlp(sent)
    matches = matcher(doc)
    if(matches is not None and len(matches)>0):
        for id, start, end in matches:
            if(str(doc[start:end]).strip() !=""):
                return True
    return False

def patterns_against_examples(file_name, patterns, examples, ids, labels):
    results = []
    for pattern in patterns:
        pattern_result = []
        working_list = expand_working_list(pattern)
        for sent in examples:
            if(check_matching(sent, working_list)):
                pattern_result.append(1)
            else:
                pattern_result.append(0)
        results.append(pattern_result)
    res = np.asarray(results).T
    df = pd.DataFrame(res, columns=patterns)
    df.insert(0,"sentences", examples)
    print(df.shape)
    df.insert(0,"labels", labels)

    df["id"] = ids

    df = df.set_index("id")
    df.to_csv(file_name)
    return df
def score(ground_truth, pred):
    tp = tn = fp = fn = 0
    for i in range(len(ground_truth)):
        if(ground_truth[i]==0 and int(pred[i][0])==0):
            tn += 1
        elif(ground_truth[i]==0 and int(pred[i][0])==1):
            fp += 1
        elif(ground_truth[i]==1 and int(pred[i][0])==0):
            fn += 1
        elif(ground_truth[i]==1 and int(pred[i][0])==1):
            tp += 1
    rec = tp /(tp+fn)
    prec = tp/(tp+fp)
    f = 2*(prec*rec)/(prec+rec)
    return (prec,rec,f) 

def train_linear_mode(df):    

    df_from_3 = df.iloc[:,3:]

    ins = torch.tensor(df_from_3.values)
    output = torch.tensor(df["labels"]).reshape(-1,1)

    selector = SelectKBest(chi2, k=20)

    
    

    X_new = selector.fit_transform(ins.numpy(), output.numpy())
    cols = selector.get_support(indices=True)

    smaller_df = df_from_3.iloc[:, cols]

    ins = torch.tensor(smaller_df.values)
    
    
    net = torch.nn.Linear(ins.shape[1],1, bias=False)
    sigmoid = torch.nn.Sigmoid()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    losses = []
    net.train()
    
    
    for e in range(5000):
        optimizer.zero_grad()
        o =  sigmoid.forward(net.forward(ins.float()))
        
        loss = criterion(o, output.float())
        
        losses.append(loss.sum().item())
        loss.backward()
        
        optimizer.step()
    

    pred =  sigmoid.forward(net.forward(ins.float())).detach().numpy()>0.5
    prec, rec, fscore = score(df["labels"], pred)
    
    
    # f = 2*(prec*rec)/(prec+rec)
    print(net.weight)
    print(prec, rec, fscore)
        
    #{fscore:float, prec:float, recall:float ,patterns:list[pat, weight] }
    response = dict()
    patterns =[]
    for i in cols.tolist():
        patterns.append(df_from_3.columns[i])
    response["fscore"] = fscore
    response["recall"] = rec
    response["precision"] = prec
    response["patterns"] = patterns
    response["weights"] = net.weight.detach().numpy()[0].tolist()
    return response





# labels = {"0":1, "1":1}
# file_name = dict_hash(labels)

# with open(f"../cache/{file_name}", "r") as file:
#     patterns = json.load(file)

# df = patterns_against_examples(file_name=file_name,patterns=list(patterns.keys()), examples=["This particular location has a good check in deal.","some of the items they sale here are a bit over priced but if you don't mind paying a bit extra this is the place to go."], ids=labels.keys(), labels=labels.values())
# train_linear_mode(df=df)