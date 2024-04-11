# Text_to_sign

## General info
The project is carring out as part of an engineering project and the "GEST" science club. The main goal is to build model to translation between polish
language and polish sign language. The sign language is represent like a gloss, we don't generate any real signs from data so far. To implement the solution, I use pretrained model from huggingface and fine-tune it. I used a model Helsinki-NLP/opus-mt-pl-en. 

## Setup
First step is to download and clone git repository:

git clone url_to_github_repo

Next we want to create a virtual enviroment and download all libraries. I will use venv library. On Windows:

python -m venv venv\
venv\Scripts\activate\
pip install -r req.txt

Second step is to create jupyter kernel from virtual enviroment.

ipython kernel install --user --name=venv_kernel

And then we can add virtual enviroment and kernel to for example a Visual Studio Code. Before we can use code, please run this scripts:
* preprocess_data.py
* load_model.py

## Technologies
* Python 
* Neptune
* Odfpy
* Tensorflow
* Transfomers
* Pandas

## Status
The project is in progress 
