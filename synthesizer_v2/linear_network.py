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


def train_linear_mode(df, price):
    inputs = df.iloc[:,3:].values
    outs = df["labels"].values


    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(inputs, outs)
    cols = selector.get_support(indices=True)
    # cols = [x for x in range(inputs.shape[1])]
    
    smaller_inputs = np.take(inputs, cols, axis=1)


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
    for e in range(500):
        optimizer.zero_grad()
        o =  sigmoid.forward(net.forward(ins.float()))
            
        loss = criterion(o, output.float())
            
        losses.append(loss.sum().item())
        loss.backward()
            
        optimizer.step()


    pred =  sigmoid.forward(net.forward(ins.float())).detach().numpy()>0.5

    labeled_prf = precision_recall_fscore_support(outs, pred, average="binary")

    selected_patterns = np.take(df.columns.values,[x+3 for x in cols] )
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

    overall_prf = precision_recall_fscore_support(entire_dataset_outs, overall_pred, average="binary")

    response = dict()


    patterns =[]
    for i in range(len(cols)):
        temp = dict()
        prf = precision_recall_fscore_support(output, df.iloc[:, cols[i]+3], average="binary" )
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







    



