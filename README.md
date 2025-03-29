# Text_to_sign
# Table of content
- [General info](#General-info)
- [Setup](#Setup)
- [Data](#Data)
- [Experiments](#Experiments)
- [Technologies](#Technologies)
- [Future improvements](#Future-improvements)
- [Status](#Status)

## General info
The project is carring out as part of an engineering project and the "GEST" science club. The main goal is to build model to translation between polish
language and polish sign language. The sign language is represent like a gloss, we don't generate any real signs from data so far. To implement the solution, I use pretrained model from huggingface and fine-tune it. I used a model Helsinki-NLP/opus-mt-pl-en.

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

Then you can run PT-notebook.ipynb or TF-notebook.ipynb cell by cell or in different way. For example if you have a trained model, you can load libraries and data and go to the test phase. The TF-notebook.ipynb won't be update in future, because I did a transition from TensorFlow to PyTorch. I also recently developed some scripts, which you can use to test models, conduct some hyperparameters tuning or visualize hyperparameter search spaces.

## Data
I started this project with small dataset (about 200 examples). Now I have above 2000 examples (deduplicated ones). I did two step to get more data. First was about two take a picture of the books with translation and then ocr it. Due to this stage I got about 1000 examples. Then I used OpenAI to create synonims for so far gathered data. I created a system prompt to steer model into good direction. I manual check all examples (some wasnt correct - I removed them). Thanks to this stage I doubled data in my dataset. Further I wil again do this step to extend my dataset.

## Experiments
All hyperparameters experiments was conducted with Bayesian optimization.
So far I mainly focused on experiments with optimizers type (SGD and ADAM), betas values, learning rate values and momentum. The table below depict findings:
| BLEU | Learning Rate | Optimizer | Momentum | Beta1  | Beta2   |
|------|--------------|-----------|----------|--------|---------|
| 0.607 | 0.000296641 | Adam      | -        | 0.628548 | 0.99944  |
| 0.604 | 0.000315742 | Adam      | -        | 0.633138 | 0.97921  |
| 0.599 | 0.000293561 | Adam      | -        | 0.543258 | 0.868566 |
| 0.598 | 0.000280291 | Adam      | -        | 0.692598 | 0.999911 |
| 0.597 | 0.000296962 | Adam      | -        | 0.564427 | 0.863337 |
| 0.588 | 0.000266087 | Adam      | -        | 0.696379 | 0.989025 |
| 0.585 | 0.000278166 | Adam      | -        | 0.693711 | 0.997883 |
| 0.584 | 0.000358374 | Adam      | -        | 0.788164 | 0.696034 |
| 0.583 | 0.00035462  | Adam      | -        | 0.427493 | 0.862276 |
| 0.582 | 0.000262948 | Adam      | -        | 0.545015 | 0.988572 |

**Testing on one category train on rest:**

*Augmented dataset, best params:*
1)	Policja - 0.438,
2)	Ratownictwo - 0.097,
3)	Urzad – 0,
4)	Urzędy - 0.365,
5)	ZUS - 0.032

Average result (BLEU): 0.1864

*Original dataset, best params:*
1)	Policja – 0,
2)	Ratownictwo - 0.009,
3)	Urzad – 0,
4)	Urzędy - 0.16,
5)	ZUS - 0.016.

Average result (BLEU): 0.037

**Cross-validation inner each category:**

*Augmented dataset, best params, 10 cross validation, batch size 64:*
1)	Policja – 0.708,
2)	Ratownictwo - 0.839,
3)	Urzad – 0.074,
4)	Urzędy - 0.704,
5)	ZUS - 0.726.

*Original dataset, best params, 10 cross validation, batch size 64:*
1)	Policja – 0.244,
2)	Ratownictwo - 0.571,
3)	Urzad – 0.024,
4)	Urzędy - 0.331,
5)	ZUS - 0.284.

*Augmented dataset, best params, 10 cross validation, batch size 32:*
1)	Policja – 0.676,
2)	Ratownictwo - 0.838,
3)	Urzad – 0.098,
4)	Urzędy - 0.723,
5)	ZUS - 0.702.

*Original dataset, best params, 10 cross validation, batch size 32:*
1)	Policja – 0.226,
2)	Ratownictwo - 0.569,
3)	Urzad – 0,
4)	Urzędy - 0.332,
5)	ZUS - 0.31.

## Technologies
* Python 
* Neptune
* Odfpy
* Tensorflow
* Transfomers
* Pandas
* PyTorch
* Optuna
* Matplotlib
* Numpy
* Scipy

## Future improvements
* compare our approach on different dataset
* create simple app with FastAPI and maybe Streamlit

## Status
The project is in progress.
