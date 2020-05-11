# NLP Adversarial Attack

Course Project for [COMS 6998-10: Robustness and Security in ML Systems](http://www.cs.columbia.edu/~junfeng/20sp-e6998/), Spring 2020 at Columbia University.

Group members:
- Weifan Jiang (wj2301)
- Haoxuan Wu (hw2754)

## Dependencies

Please install dependencies including [stanford NLP](https://pypi.org/project/stanfordnlp/) with English models, [keras-preprocessing](https://pypi.org/project/Keras-Preprocessing/), [google cloud translation API](https://cloud.google.com/translate/docs/reference/libraries/v2/python) (may need to set up an account and provide credit card information), and [tensorflow](https://www.tensorflow.org/) version 1 (our project is incompatible with version 2 or greater), etc.

Other dependencies can be downloaded following prompt.

## Input data

Please download this [zip file](https://drive.google.com/open?id=19FCkbg9IpbdFshBWNV_hXCWswsBF6bkk) from google drive (Columbia account required), and unzip it in the root of repository.

## Usage

Our implementation of the Sentence-level genetic algorithm with salency analysis, along with other helper functions is in `sentence_level_genetic_attack.py`. Please change the envrionmental variables in the script accordingly (dataset name, google cloud API credentials, etc.).

A demostration on how to use our attack is in `sentence_level_genetic_attack_demo.ipynb`. The example model used is a __WordCNN__ trained with __imdb__ dataset.

## Credit

The pretrained models and preprocessing of input data are from [Alzantot Et Al.](https://www.aclweb.org/anthology/D18-1316/). Our forked version of their original repo is [here](https://github.com/weifanjiang/nlp_adversarial_examples). 
