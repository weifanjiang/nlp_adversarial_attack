# Weifan Jiang, wj2301@columbia.edu
# Haoxuan Wu, hw2754@columbia.edu
# This file contains main implementation of our genetic attack algorithm

from neural_networks import word_cnn, char_cnn, bd_lstm, lstm
import os
from read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from word_level_process import word_process, get_tokenizer, text_to_vector_for_all
from config import config
from keras.preprocessing import sequence
import numpy as np
import random
import pickle

############################# ENVIRONMENTAL VARIABLES #################################

import stanfordnlp
# uncomment if needed
# stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()

DATASET = "imdb"  # change if needed

import os

GOOGLE_LICENSE_FILELOC = "/Users/weifanjiang/Documents/Personal/My Project-1e7426894fe6.json" # change to correct path

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= GOOGLE_LICENSE_FILELOC
########################################################################################

def sentence_list(doc):
    """
    Returns a list of sentences from the given stanfordnlp instance.

    A stanford nlp instance can be constructed by
    > xi_text = "Good morning."
    > doc = nlp(xi_text)
    """
    sentences = []
    for words in doc.sentences:
        sentence = words.words[0].text
        for word in words.words[1:]:
            if word.upos != 'PUNCT' and not word.text.startswith('\''):
                sentence += ' '
            sentence += word.text
        sentences.append(sentence)
    return sentences

def predict_str(model, s):
    """
    Predict confidence score if string input
    """
    maxlen = config.word_max_len[DATASET]
    tokenizer = get_tokenizer(DATASET)
    s_seq = tokenizer.texts_to_sequences([s])
    s_seq = sequence.pad_sequences(s_seq, maxlen=maxlen, padding='post', truncating='post')
    s_sep = s_seq[0]
    return model.predict(s_seq)[0]

def predict_sentences(model, sentences):
    """
    Predict confidence score if list of strings input
    """
    return predict_str(model, ' '.join(sentences))

def sentence_saliency(model, sentences, label):
    """
    Compute saliency scores of a list of sentences.
    """
    true_pred = predict_str(model, ' '.join(sentences))

    # idx: index of confidence score of correct label
    if label[0] == 1:
        idx = 0
    else:
        idx = 1

    scores = []
    for i in range(len(sentences)):
        x_hat = ' '.join(sentences[0:i] + sentences[i+1:])
        scores.append(true_pred[idx] - predict_str(model, x_hat)[idx])
    
    return np.array(scores)

def softmax(x, determinism = 10):
    """
    Softmax function with parameter specifying determinism for difference.
    """
    softmax = np.exp(np.multiply(determinism, x))
    softmax /= np.sum(softmax)
    return softmax

def test_google_license():
    """
    Test google cloud API set-up.
    Return true upon successful.
    """
    from google.cloud import storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= GOOGLE_LICENSE_FILELOC

    try:
        # If you don't specify credentials when constructing the client, the
        # client library will look for credentials in the environment.
        storage_client = storage.Client()

        # Make an authenticated API request
        buckets = list(storage_client.list_buckets())
        return True
    except:
        return False

def back_translation(s_in, language='ko', require_mid=False):
    """
    Perform back translation to paraphrase an English input.

    s_in(str): input English sentence
    language(str): middle language, should be a two letter code.
    require_mid(bool): if True, returns middle translation result and final result. Otherwise
        return final only.
    """

    from google.cloud import translate_v2 as translate
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= GOOGLE_LICENSE_FILELOC

    translate_client = translate.Client()
    mid_result = translate_client.translate(s_in, target_language=language)['translatedText']
    if require_mid:
        mid = mid_result.replace("&#39;", "\'")
    en_result = translate_client.translate(mid_result, target_language="en")['translatedText'].replace("&#39;", "\'")
    if require_mid:
        return mid, en_result
    else:
        return en_result

def load_cache():
    """
    Load cached google translation results from pickle.
    """
    cache_file = "cache/google_translate_cache.pickle"
    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f_in:
            cache = pickle.load(f_in)
    else:
        cache = dict()
    return cache

def save_cache(cache):
    """
    Save updated cache.
    """
    cache_file = "cache/google_translate_cache.pickle"
    with open(cache_file, "wb") as f_out:
        pickle.dump(cache, f_out)

def get_all_languages():
    """
    Get all options of middle language code that can be used for middle translation.
    """
    with open("cache/all_languages.pickle", "rb") as fin:
        ret = pickle.load(fin)
    return ret

############# Implementation of Sentence-level genetic algorithm ##########################

def perturb(sentences, saliencies, cache, not_choose=None):
    """
    Random perturbation to the input list of sentences.

    params:
        sentences: list of sentences
        saliences: saliency scores of sentences
        cache: google translate cache
        not_choose: indices of sentence not to choose to perturb.
            usually previously perturbed sentences from previous iterations of Genetic Algorithm.
    
    Note it tries back translation for 20 times to get a different structured sentence from original.
    Weifan's personal credit card is on Google Cloud so he's very careful.
    """

    if not_choose is None:
        not_choose = set()

    choices = list()
    weights = list()
    for i in range(len(sentences)):
        if i not in not_choose:
            choices.append(sentences[i])
            weights.append(saliencies[i])
    weights = softmax(weights, determinism=30)
    
    choice = np.random.choice(choices, p=weights)
    all_rephrase = set()
    
    # Generate Back Translation Example
    chosen_language = set()
    all_languages = list(get_all_languages())
    rephrase = choice
    count = 0
    while (len(chosen_language) < len(all_languages)) and (rephrase == choice):
        language = np.random.choice(all_languages)
        while language in chosen_language:
            language = np.random.choice(all_languages)
        chosen_language.add(language)
        rephrase = cache.get((choice, language), None)
        if rephrase is None:
            rephrase = back_translation(choice, language=language)
            cache[(choice, language)] = rephrase
        count += 1
        if count == 20:
            raise ValueError("20 calls to API! Google has my credit card, time to save money.")
    all_rephrase.add(rephrase)

    chosen_rephrase = random.sample(all_rephrase, 1)[0]
    new_paragraph = []
    for j, sen in enumerate(sentences):
        if sen == choice:
            new_paragraph.append(chosen_rephrase)
            changed_idx = j
        else:
            new_paragraph.append(sen)
    return new_paragraph, changed_idx

def crossover(sentences1, sentences2, p1_changed, p2_changed):
    """
    Crossover two set of sentences.
    Return the crossover child, also the crossover changed indices.
    """
    child = list()
    child_idx = set()
    for i in range(len(sentences1)):
        if random.randint(0,1) == 0:
            if i in p1_changed:
                child_idx.add(i)
            child.append(sentences1[i])
        else:
            if i in p2_changed:
                child_idx.add(i)
            child.append(sentences2[i])
    return child, child_idx

def genetic(x0, y0, model, population, generation, cache = None, verbose = False):
    """
    Main implementation of sentence-level genetic attack with saliency analysis.

    params:
        x0(str): clean sample
        y0(str): clean label, either [1, 0] or [0, 1]
        model: model to attack. Should support model.predict function.
        population(int): size of each population
        generation(int): max number of generations to try Genetic Attack.
        cache(dict): Google Translate cache. Should be loaded from cache/ directory.
        verbose(bool): whether or not verbose attack.
    
    returns:
        str: if successfully generates adv. example, return adv. example as string
        None: if timed out.
    """

    if cache is None:
        cache = load_cache()
    
    if y0[0] == 0:
        y_adv = [1, 0]
    else:
        y_adv = [0, 1]
    
    doc = nlp(x0)
    sentences = sentence_list(doc)
    saliency_scores = sentence_saliency(model, sentences, y0)
    
    prob_0 = predict_str(model, x0)
    
    if verbose:
        print("clean sample's prediction: {}".format(prob_0))
    
    # We want target_idx of adv.example's prediction to be larger
    # than 0.5
    target_idx = np.argmin(prob_0)
    if verbose:
        print('target is to make index {} > 0.5'.format(target_idx))
    
    gen0 = list()
    chosen_idx = list()
    for i in range(population):
        chosen = set()
        sample, idx = perturb(sentences, saliency_scores, cache)
        gen0.append(sample)
        chosen.add(idx)
        chosen_idx.append(chosen)
    
    curr_gen = gen0
    for i in range(generation):
        
        if verbose:
            print('generation {}'.format(i + 1))
        
        sample_weight = list()
        for j, sample in enumerate(curr_gen):
            sample_pred = predict_sentences(model, sample)
            if verbose:
                print("population {} pred: {}, changed idx: {}".format(j, sample_pred, chosen_idx[j]))
            if sample_pred[target_idx] > 0.5:
                if verbose:
                    print('successful adv. example found!')
                save_cache(cache)
                return ' '.join(sample)
            else:
                sample_weight.append(sample_pred[target_idx])
        sample_weight = softmax(np.array(sample_weight))
        if verbose:
            print('population with fitness scores: {}'.format(sample_weight))
        
        next_gen = list()
        next_chosen = list()
        for j in range(population):
            idx_list = list(range(population))
            p1 = np.random.choice(idx_list, p=sample_weight)
            p2 = np.random.choice(idx_list, p=sample_weight)
            if verbose:
                print("child {} generated with parents {} and {}".format(j, p1, p2))
            child, child_change = crossover(curr_gen[p1], curr_gen[p2], chosen_idx[p1], chosen_idx[p2])
            saliency_scores = sentence_saliency(model, child, y0)
            child_mutate, change_idx = perturb(sentences, saliency_scores, cache, child_change)
            next_gen.append(child_mutate)
            child_change.add(change_idx)
            next_chosen.append(child_change)
        curr_gen = next_gen
        chosen_idx = next_chosen

    save_cache(cache)
    return None