# Text_to_sign

## General info
The project is carring out as part of an engineering project and the "GEST" science club. The main goal is to build model to translation between polish
language and polish sign language. The sign language is represent like a gloss, we don't generate any real signs from data so far. To implement the solution, I use pretrained model from huggingface and fine-tune it. I used a model Helsinki-NLP/opus-mt-pl-en. I recenlty
added a PyTorch notebook. This notebook has not finished, because I checked how much I can speed my training on GPU. In the future I will rewrite all code in PyTorch.

## Setup
First step is to download and clone git repository:
```
git clone url_to_github_repo
```
Next we want to create a virtual enviroment and download all libraries. I will use venv library. On Windows:
```
python -m venv venv
venv\Scripts\activate
pip install -r req.txt
```
Second step is to create jupyter kernel from virtual enviroment.
```
ipython kernel install --user --name=venv_kernel
```
And then we can add virtual enviroment and kernel to for example a Visual Studio Code. Before we can use code, please run this scripts:
* preprocess_data.py
* load_model.py

The last step is to create .env file with:

NEPTUNE_API_TOKEN = "your API key"\
NEPTUNE_PROJECT = "your project name"\
TF_CPP_MIN_LOG_LEVEL = "3"

Then you can run Main.ipynb notebook cell by cell or in different way. For example if you have a trained model, you can load libraries and data and go to the test phase.

## Technologies
* Python 
* Neptune
* Odfpy
* Tensorflow
* Transfomers
* Pandas
* PyTorch

## Status
The project is in progress.
