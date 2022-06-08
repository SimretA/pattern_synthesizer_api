import pandas as pd
import numpy as np
import torch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import precision_recall_fscore_support


import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")

from synthesizer.helpers import expand_working_list
import json

import warnings
warnings.filterwarnings('ignore')

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

# define df, columns, true labels
def feature_selector_old(df):
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
            fscore = precision_recall_fscore_support(labels, col_selected,  average="weighted")[2]
            collector[col] =  fscore
        
        #sort and get a pattern with high fscore
        collector = {k: v for k, v in sorted(collector.items(), key=lambda item: item[1])}
        selected_starter_pattern = list(collector.keys())[-1]
        selected_starter_series = df[selected_starter_pattern]
        patterns_selected.append(selected_starter_pattern)
        print(selected_starter_pattern, collector[selected_starter_pattern])
        
        #get rid of all correlated patterns
        corr = df.corr()
        to_drop = [c for c in corr.columns if corr[selected_starter_pattern][c] >= 0.9] #0.9 chosen at random
        
        df = df.drop(to_drop, axis=1)
        print

        #create a new df with combination of current one
        remaining_cols = df.columns.values[4:]
        for coll in remaining_cols:
            df[coll] = np.logical_or(df[coll], selected_starter_series)
        
        print(f"Finishing iteration {i} {len(remaining_cols)}")
    return patterns_selected
def train_and_report(patterns, inputs, outputs):
    #Change numpy inputs to tensors 
    outputs = torch.tensor(outputs).reshape(-1,1)
    inputs = torch.tensor(inputs)

    #train the linear layer for 100 iterations
    #100 chosen at random TODO see what a good number is for iteration

    net = torch.nn.Linear(inputs.shape[1],1, bias=False)
    sigmoid = torch.nn.Sigmoid()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    losses = []
    net.train()
    for e in range(50):
        optimizer.zero_grad()
        o =  sigmoid.forward(net.forward(inputs.float()))
            
        loss = criterion(o, outputs.float())
            
        losses.append(loss.sum().item())
        loss.backward()
            
        optimizer.step()
    
    pred =  sigmoid.forward(net.forward(inputs.float())).detach().numpy()>0.5
    labeled_prf = precision_recall_fscore_support(outputs, pred, average="weighted")

    fscore = labeled_prf[2]
    # print(f"{patterns}, {fscore}")


    return fscore

# define df, columns, true labels
def feature_selector(df):

    positive_examples = df[df['labels']==1]['sentences'].values
    negative_examples = df[df['labels']==0]['sentences'].values

    print(f"==================================Start of Feature Selection===========================================")
    labels = df['labels']
    jj = 0
    ### Controller variables
    patterns_selected = []
    highest_fscore = "0.0"
    df_subset = pd.DataFrame()
    remaining_cols = df.columns.values[4:]


    outputs = df["labels"].values
    while len(patterns_selected)<10 and len(remaining_cols)>0:
        jj += 1
        print(f"Starting iteration {jj} {len(remaining_cols)}")
        #first calculate the fscore
        collector = {}
        local_max_fscore = "0.0"
        for col in remaining_cols:
            col_selected = df[col].astype('int64')
            current_patterns = patterns_selected+[col]
            current_df = pd.concat([df_subset, col_selected], axis=1)
            inputs = current_df.values
            
            # fscore = precision_recall_fscore_support(labels, col_selected,  average="binary")[2]
            fscore = train_and_report(current_patterns, inputs, outputs)
            
                
            exists = str(fscore) in collector
            if(exists):
                collector[str(fscore)].append(col)
                
            else:
                collector[str(fscore)] = [col]
        #sort and get a pattern with high fscore
        selected_starter_pattern = list(collector.values())[-1]
        collector = {k: v for k, v in sorted(collector.items(), key=lambda item: item[0])}
        current_fscore = list(collector.keys())[-1]

        if(current_fscore>highest_fscore):
            highest_fscore = current_fscore
        else:
            # print(f"{highest_fscore} {current_fscore}, {selected_starter_pattern}")
            break
        selected_starter_pattern = list(collector.values())[-1]

        #Group the correlated ones and pick the shortest
        rowss = df[selected_starter_pattern]
        correlation = rowss.corr()
        correlation.loc[:,:] =  np.tril(correlation, k=-1)
        cor = correlation.stack()
        ones = cor[cor >=0.8].reset_index().loc[:,['level_0','level_1']]
        ones = ones.query('level_0 not in level_1')
        grps = list(ones.groupby('level_0').groups.keys())
        colls = []
        #NOUN
        #NOUN+NUX+ADJ
        for i in grps:
            groups = ones[ones["level_0"]==i].values
            set_maker = []
            for patterns in groups:
                set_maker += patterns.tolist()
            colls.append(sorted(set_maker, key=len)[0])
            
        for selected_starter_pattern in colls:
            patterns_selected.append(selected_starter_pattern)
            df_subset[selected_starter_pattern] = df[selected_starter_pattern].astype('int64')
            try:
                selected_starter_series = df[selected_starter_pattern][0]
                
                corr = df.corr()
                to_drop = [c for c in corr.columns if corr[selected_starter_pattern][c] >= 0.8] #0.8 chosen at random
                df = df.drop(to_drop, axis=1)

                #create a new df with combination of current one
                remaining_cols = df.columns.values[4:]
                for collumn in remaining_cols:
                    df[collumn] = np.logical_or(df[collumn], selected_starter_series)
            except:
                print("We already removed ", selected_starter_pattern)
            for coll in remaining_cols:
                df[coll] = np.logical_or(df[coll], selected_starter_series)
        
        print(f"Finishing iteration {jj} {len(remaining_cols)}, --- {patterns_selected}, {highest_fscore}")
    
    print(f"---------------------------Summary---------------------------")
    print(f"Patterns {patterns_selected}")
    print(f"Positive examples \n{positive_examples}")
    print(f"Negative examples \n{negative_examples}")

    print(f"==================================End of Feature Selection===========================================")
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

    labeled_prf = precision_recall_fscore_support(outs, pred, average="weighted")

    # selected_patterns = np.take(df.columns.values,[x+3 for x in cols] )
    selected_patterns = cols
    selected_working_list = []
    for pattern in selected_patterns:
        selected_working_list.append(expand_working_list(pattern))

    running_result = []

    for sentence in price["example"].values:
        temp = []
        
        for i in range(len(selected_working_list)):
            temp.append(int(check_matching(sentence, selected_working_list[i])))
        running_result.append(temp)

    entire_dataset_ins = torch.Tensor(running_result)

    entire_dataset_outs = torch.Tensor(price["positive"].values).reshape(-1,1)

    print(entire_dataset_ins.shape)

    overall_prob = sigmoid.forward(net.forward(entire_dataset_ins.float())) 

    overall_pred = overall_prob.detach().numpy()>0.5

    overall_prf = precision_recall_fscore_support(entire_dataset_outs, overall_pred, average="weighted")

    response = dict()


    patterns =[]
    for i in range(len(cols)):
        temp = dict()
        prf = precision_recall_fscore_support(output, df[cols[i]], average="weighted" )
        temp["pattern"] = selected_patterns[i]
        temp["precision"] = prf[0]
        temp["recall"] = prf[1]
        temp["fscore"] = prf[2]

        patterns.append(temp)

    response["fscore"] = labeled_prf[2]
    response["recall"] = labeled_prf[1]
    response["precision"] = labeled_prf[0]


    response["overall_fscore"] = overall_prf[2]
    response["overall_recall"] = overall_prf[1]
    response["overall_precision"] = overall_prf[0]
    


    response["patterns"] = patterns
    response["weights"] = net.weight.detach().numpy()[0].tolist()

    response["scores"] = [x[0] for x in overall_prob.tolist()]

    return [net, response]







    



