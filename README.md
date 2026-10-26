# finetuning-google-mt5
Finetuning Google MT5 using HF-Transformers. 

Install requirements:
`pip install -r requirements.txt` into a virtual environment.

Train model using:
`python trainer.py`

Benchmark predictions with trained model using:
`python eval.py`

Don't forget to put your Groq API key into the environment variables before making LLM calls.