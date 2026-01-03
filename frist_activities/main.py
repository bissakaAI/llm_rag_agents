from fastapi import FastAPI,File,UploadFile
from dotenv import load_dotenv
import os 
from pydantic import BaseModel, typing
import uvicorn

import torch
import torch.nn as nn
import torch.optim as optm
from torch.utils.data import DataLoader
import io
from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()
api=os.getenv('openai_key')
print(api)


app= FastAPI(title='Hair Classifier App',version='1.0.0')

from fastapi.middleware.cors import CORSMiddleware



@app.get("/")
def landingpage():
    return {"message":"Welcome to the Hair Classifier App"}



@app.post("/feedback")
def feedback(userinput):

    userinput= input("Rate the prediction app")
    prompt= f"""Determine the sentiment of the following sentences:
    text: "I love this product! it works perfectly" -> classification: Positive
    text: "i dont really know what to say about this restaurant, the food was not bad and i wouldnt say it was great.it was ok"-> classification: Neutral
    text: "The movie was boring,I was sleeping 10 minutes into the movie" ->classification: Negative

    Now analyse this sentence:
    {userinput} """

    client = OpenAI(api_key=api)
    response= client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}],temperature=0)

    print(response.choices[0].message.content)

if __name__ == '__main__':
    uvicorn.run(app,host=os.getenv("host"),port=int(os.getenv("port")))
