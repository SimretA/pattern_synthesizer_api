from fastapi import FastAPI
from synthesizer.api_helper import *
from synthesizer.penality_based_threaded import Synthesizer 
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
app = FastAPI()
api_helper = APIHelper()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# @app.get("/patterns")
# async def root():
#     synthh = Synthesizer(positive_examples = "examples/small", negative_examples = "examples/small_neg")
#     return synthh.find_patters(outfile="40pos_33neg_price_1thresh")
    # return {"message": "Hello World"}



# @app.get("/dataset")
# async def get_examples():
    
#     return api_helper.get_dataset()

@app.get("/dataset")
async def get_labeled_examples():
    
    return api_helper.get_labeled_dataset()

@app.post("/label/{id}/{label}")
async def label_example(id:str, label: int):
    # print("got it")
    return api_helper.labeler(id, label)

@app.post("/clear")
async def clear_labeling():
    return api_helper.clear_label()

@app.get("/combinedpatterns")
async def cpmbinedpatterns():
    return api_helper.resyntesize()

@app.get("/patterns")
async def patterns():
    return api_helper.all_patterns()

@app.get("/testing_cache")
async def testing_patterns():
    return api_helper.testing_cache()


def main():
    synthh = Synthesizer(positive_examples = "examples/price_big", negative_examples = "examples/not_price_big")
    print(synthh.find_patters(outfile="small_thresh"))
# main() pid 31616