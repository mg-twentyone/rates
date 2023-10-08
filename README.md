# Satsconverter

A Basic Bitcoin to Fiat converter with price feeds from Coindesk.

# FastAPI + Vercel

This app runs on FastAPI on Vercel

## To install

```sh
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

## To run this app locally

```sh
uvicorn src.app:app --reload
```

Works up to python version 3.9.15
for current requirements.txt

Your application is now available at `http://localhost:8000`.

# Attribution 

original source code from https://github.com/bitkarrot/satsconverter
