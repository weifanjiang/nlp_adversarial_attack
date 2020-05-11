# NLP Adversarial Attack

Course Project for [COMS 6998-10: Robustness and Security in ML Systems](http://www.cs.columbia.edu/~junfeng/20sp-e6998/), Spring 2020 at Columbia University.

Group members:
- Weifan Jiang (wj2301)
- Haoxuan Wu (hw2754)

## Set up

Please install dependencies including [stanford NLP](https://pypi.org/project/stanfordnlp/) with English models, [keras-preprocessing](https://pypi.org/project/Keras-Preprocessing/), and [tensorflow](https://www.tensorflow.org/), etc.

Note that this project requires tensorflow version 1 and is incompatible with version 2 and/or above.

## Usage

Our implementation of the Sentence-level genetic algorithm with sentence-level perturbation, along with other helper functions is in `sentence_level_genetic_attack.py`. Please change the envrionmental variables in the script accordingly (dataset name, google cloud API credentials, etc.).

A demostration on how to use our attack is in `sentence_level_genetic_attack_demo.ipynb`. The example model is trained with __WordCNN__ and __imdb__ dataset.
