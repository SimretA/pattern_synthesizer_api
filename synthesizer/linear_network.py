import pandas as pd
import numpy as np
import torch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.metrics import precision_recall_fscore_support


import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")

from synthesizer.helpers import expand_working_list



import random


def check_matching(sent, working_list, explain=False):
    matcher = Matcher(nlp.vocab)
    for index, patterns in enumerate(working_list):
        matcher.add(f"rule{index}", [patterns])
    doc = nlp(sent)
    matches = matcher(doc)
    if(matches is not None and len(matches)>0):
        for id, start, end in matches:
            if(str(doc[start:end]).strip() !=""):
                if(explain):
                    return (True, str(doc[start:end]).strip())
                return True
    if(explain):
        return (False, "")
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


# define df, columns, true labels
def feature_selector(df):
    current_fscore = -999
    remaining_cols = df.columns.values[4:]
    labels = df['labels']
    patterns_selected = []
    i = 0
    while len(patterns_selected)<10 and len(remaining_cols)>0:
        i += 1
        print(f"Starting iteration {i} {len(remaining_cols)}")
        #first calculate the fscore
        collector = {}
        for col in remaining_cols:
            col_selected = df[col]
            fscore = precision_recall_fscore_support(labels, col_selected,  average="binary")[2]
            collector[col] =  fscore
        
        #sort and get a pattern with high fscore
        collector = {k: v for k, v in sorted(collector.items(), key=lambda item: item[1])}
        
        selected_starter_pattern = list(collector.keys())[-1]
        
        selected_fscore = list(collector.values())[-1]
        if(selected_fscore<current_fscore):
            break
        current_fscore = selected_fscore
        selected_starter_series = df[selected_starter_pattern]
        patterns_selected.append(selected_starter_pattern)
        print(selected_starter_pattern, collector[selected_starter_pattern])
        
        #get rid of all correlated patterns
        corr = df.corr()
        
        to_drop = [c for c in corr.columns if corr[selected_starter_pattern][c] >= 0.8 or corr[selected_starter_pattern][c] <= 0] #0.9 chosen at random
        to_drop.append(selected_starter_pattern)
        
        df = df.drop(to_drop, axis=1)

        #create a new df with combination of current one
        remaining_cols = df.columns.values[4:]
        # for coll in remaining_cols:
        #     df[coll] = np.logical_or(df[coll], selected_starter_series)
        
        print(f"Finishing iteration {i} {len(remaining_cols)}")
    return patterns_selected


def train_linear_mode(df, price):
    
    outs = df["labels"].values
    

    cols = feature_selector(df)
   


    # smaller_inputs = np.take(inputs, cols, axis=1)
    smaller_inputs =  df[cols].values


    ins = torch.tensor(smaller_inputs)
    
    # ins = torch.tensor(inputs)

    output = torch.tensor(outs).reshape(-1,1)


    net = torch.nn.Linear(ins.shape[1],1, bias=False)
    sigmoid = torch.nn.Sigmoid()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    losses = []
    net.train()
    print("training ...")
    for e in range(100):
        optimizer.zero_grad()
        o =  sigmoid.forward(net.forward(ins.float()))
            
        loss = criterion(o, output.float())
            
        losses.append(loss.sum().item())
        loss.backward()
            
        optimizer.step()


    pred =  sigmoid.forward(net.forward(ins.float())).detach().numpy()>0.5

    labeled_prf = precision_recall_fscore_support(outs, pred, average="binary")

    selected_patterns = cols
    selected_working_list = []
    matched_parts = {}
    for pattern in selected_patterns:
        selected_working_list.append(expand_working_list(pattern))
        matched_parts[pattern] = {}

    running_result = []

    for sentence,id in zip(price["example"].values, price["id"].values):
        temp = []
        
        for i in range(len(selected_working_list)):
            it_matched = check_matching(sentence, selected_working_list[i], explain=True)
            temp.append(int(it_matched[0]))
            matched_parts[selected_patterns[i]][id] = it_matched[1] #.append({int(id): it_matched[1]})

        running_result.append(temp)

    entire_dataset_ins = torch.Tensor(running_result)

    entire_dataset_outs = torch.Tensor(price["positive"].values).reshape(-1,1)
    

    print(entire_dataset_ins.shape)

    overall_prob = sigmoid.forward(net.forward(entire_dataset_ins.float())) 

    overall_pred = overall_prob.detach().numpy()>0.5
    ids = price['id'].values.tolist()

    overall_prf = precision_recall_fscore_support(entire_dataset_outs, overall_pred, average="binary")

    response = dict()

    response["explanation"] = matched_parts


    patterns =[]

    weights = net.weight.detach().numpy()[0].tolist()
    for i in range(len(cols)):
        temp = dict()
        prf = precision_recall_fscore_support(output, df[ cols[i]], average="binary" )
        temp["pattern"] = selected_patterns[i]
        temp["precision"] = prf[0]
        temp["recall"] = prf[1]
        temp["fscore"] = prf[2]
        temp["weight"] = weights[i]

        patterns.append(temp)

    response["fscore"] = labeled_prf[2]
    response["recall"] = labeled_prf[1]
    response["precision"] = labeled_prf[0]


    response["overall_fscore"] = overall_prf[2]
    response["overall_recall"] = overall_prf[1]
    response["overall_precision"] = overall_prf[0]


    response["patterns"] = patterns
    response["weights"] = net.weight.detach().numpy()[0].tolist()



    response["scores"] = { x:y[0] for x,y in zip(ids, overall_prob.tolist()) }

    # response["scores"] = { x:random.uniform(0, 1) for x,y in zip(ids, overall_prob.tolist()) }


    return response







    



