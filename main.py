from unicodedata import name
from unittest import result
from fastapi import FastAPI
from synthesizer.api_helper import *
from synthesizer_v2.api_helper import *
from synthesizer.penality_based_threaded import Synthesizer 
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
import asyncio
import time
from pydantic import BaseModel



import pandas as pd

import random
from synthesizer_v2.linear_network import feature_selector
import torch
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

executor = ProcessPoolExecutor()
loop = asyncio.get_event_loop()


app = FastAPI()
api_helper = APIHelper()
api_helper2 = APIHelper2()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    depth: int
    rewardThreshold: float
    penalityThreshold: float
    featureSelector: int





####v2 endpoints
@app.get("/v2/dataset")
async def get_labeled_examples():
    
    return api_helper2.get_labeled_dataset()

@app.post("/v2/theme/{theme}")
async def get_labeled_examples(theme:str):
    
    return api_helper2.add_theme(theme)


@app.post("/v2/label/{id}/{label}")
async def label_example(id:str, label: str):
    # print("got it")
    return api_helper2.labeler(id, label)

@app.post("/v2/clear")
async def clear_labeling():
    return api_helper2.clear_label()

@app.get("/v2/combinedpatterns")
async def combinedpatterns():
    return api_helper2.resyntesize()

@app.get("/v2/patterns")
async def patterns():
    return api_helper2.all_patterns()

@app.get("/v2/next_dataset")
async def next_dataset():
    return api_helper2.get_next_data()

@app.get("/v2/testing_cache")
async def testing_patterns():
    return api_helper2.testing_cache()

@app.get("/v2/testing_ordering")
async def testing_ordering():
    return api_helper2.testing_label_ordering()






threadpool = {}

###v1 endpoints
@app.get("/dataset")
async def get_labeled_examples():
    
    return api_helper.get_labeled_dataset()

# @app.post("/label/{id}/{label}")
# async def label_example(id:str, label: int):
#     # if asyncio.Task:
#     #     for task in asyncio.all_tasks():
#     #         print("task", task)
#     future1 = loop.run_in_executor(None, api_helper.labeler, id, label)
#     res = await future1
#     # results = await api_helper.labeler(id, label)
#     return res

@app.post("/phrase/{phrase}/{label}")
async def label_by_phrase(phrase:str, label: int):
    # print("got it")

    results = await loop.run_in_executor(executor, api_helper.label_by_phrase, phrase, label)
    return results

@app.post("/clear")
async def clear_labeling():
    return api_helper.clear_label()

@app.get("/combinedpatterns")
async def combinedpatterns():
    results = await loop.run_in_executor(executor, api_helper.resyntesize)
    api_helper.results = results
    return results



@app.get("/testing_cache")
async def testing_patterns():
    return api_helper.testing_cache_new()

@app.get("/test/{iteration}/{annotation}")
async def test(iteration:int, annotation: int, body:Item):
    print(body)
    start =  time.time()
    results = api_helper.run_test(iteration, annotation, depth= body.depth, rewardThreshold=body.rewardThreshold, penalityThreshold=body.penalityThreshold)
    end = time.time()
    print(results)
    results[0]['time'] = end-start

    return results

@app.get("/themes")
async def get_themes():
    return api_helper.get_themes()

@app.get("/selected_theme")
async def get_selected_theme():
    return api_helper.get_selected_theme()

@app.post("/set_theme/{theme}")
async def set_theme(theme:str):
    return (theme,api_helper.set_theme(theme))

@app.get("/related_examples/{id}")
async def get_related_examples(id:str):
    results = await loop.run_in_executor(executor, api_helper.get_related, id)
    return results
@app.get("/explain/{pattern}")
async def explain_pattern(pattern:str):
    results = await loop.run_in_executor(executor, api_helper.explain_pattern, pattern)
    return results

def main():
    synthh = Synthesizer(positive_examples = "examples/price_big", negative_examples = "examples/not_price_big")
    print(synthh.find_patters(outfile="small_thresh"))
# main() pid 31616

######################################################################################################################
@app.post("/delete_label/{id}/{label}")
async def delete_label(id:str, label: str):
    # if asyncio.Task:
    #     for task in asyncio.all_tasks():
    #         print("task", task)
    future1 = loop.run_in_executor(None, api_helper.delete_label, id, label)
    res = await future1
    # results = await api_helper.labeler(id, label)
    return res

@app.post("/label/{id}/{label}")
async def label_example(id:str, label: str):
    # if asyncio.Task:
    #     for task in asyncio.all_tasks():
    #         print("task", task)
    future1 = loop.run_in_executor(None, api_helper.label_element, id, label)
    res = await future1
    # results = await api_helper.labeler(id, label)
    return res

@app.get("/patterns")
async def patterns():
    results = await loop.run_in_executor(executor, api_helper.synthesize_patterns)
    return results