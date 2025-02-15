import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query
from typing import List
import uvicorn

app=FastAPI()

class processed_text(BaseModel):
    input_text:str
    keywords: List[str]
    summary: str
    sentiment: str

class text_input(BaseModel):
    text:str

summary_tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summary_model=AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

ner_tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model=AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

sentiment_tokenizer=AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model=AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

history=[]

@app.post("/process",response_model=processed_text)
def process_text(input_data: text_input):
    text=input_data.text.strip()

    if not text:
        raise HTTPException(response_code=400,detail="Text can't be empty")
    
    if len(text)>1000:
        raise HTTPException(response_model=400,detail="Text can't exceed more than 1000")
    
    #summarize the input text
    inputs=summary_tokenizer(text,return_tensors='pt',truncation=True)
    summary_ids=summary_model.generate(inputs.input_ids,max_length=50,min_length=10,do_sample=False)
    summary=summary_tokenizer.decode(summary_ids[0],skip_special_tokens=True)

    #Extract keywords and named entities
    tokens=ner_tokenizer(text,return_tensors='pt',truncation=True)
    outputs=ner_model(**tokens)
    token_ids=tokens.input_ids.squeeze(0).tolist()
    predictions=torch.argmax(outputs.logits,dim=2).squeeze(0).tolist()
    keywords=[ner_tokenizer.convert_ids_to_tokens(token_id) for token_id,pred in zip(token_ids,predictions) if pred!=0]
    keywords=list(set(keywords))

    #sentiment analysis
    inputs=sentiment_tokenizer(text,return_tensors='pt',truncation=True)
    outputs=sentiment_model(**inputs)
    sentiment_score=torch.argmax(outputs.logits,dim=1).item()
    sentiment_mapping={0:"Negative",1:"Neutral",2:"Postive"}
    sentiment=sentiment_mapping.get(sentiment_score,"Unknown")

    #store in history
    processed_entry={
        "input_text":text,
        "keywords":keywords,
        "summary":summary,
        "sentiment":sentiment
    }

    history.append(processed_entry)

    return processed_entry

@app.get("/history",response_model=List[processed_text])
def get_history():
    return history

@app.get('/search',response_model=List[processed_text])
def search_history(query:str=Query(...,min_length=1)):
    return [entry for entry in history if query.lower() in entry["input_text"].lower() or query in entry["keywords"]]

@app.delete('/clear')
def clear_history():
    history.clear()
    return {"message":"History cleared successsfully"}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)

    

