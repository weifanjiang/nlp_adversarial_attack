#!/usr/bin/env python
import sys

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import gensim
import numpy as np
import string

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]
stop_words = stopwords.words('english')



class Word2VecSubst(object):
        
    def __init__(self, filename = 'C:/Users/wu/Desktop/Columbia CS HW/NLP/HW4/hw4/hw4_files/GoogleNews-vectors-negative300.bin.gz'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        self.lemmatizer = WordNetLemmatizer()
        self.skip_list = ['is','are','were','was','am']
    def synonym_generation(self, sentences):
        new_sentences = nltk.sent_tokenize(sentences)
        result = []
        
        for ns in new_sentences:
            word_tk = nltk.word_tokenize(ns)
            # use tokenized sentence for pos tag
            word_tg = nltk.pos_tag(word_tk)
            for i in range(0,len(word_tg)):
                tg_tup = word_tg[i]
                # tg_tup (word,pos)
                if tg_tup[0] in self.model.vocab and tg_tup[0] not in self.skip_list and self.check_pos(tg_tup[1]):
                    word_tk[i] = self.predict_nearest_sys(tg_tup[0],tg_tup[1])
            for w in word_tk:
                result.append(w)
                result.append(' ')
        return "".join(result)
    
    
    def predict_nearest_sys(self, word, pos):
        possible_synonyms = self.get_candidates(word, pos)
        target_word = word
        max_similarity = -1
        max_synonym = target_word
        for synonym in possible_synonyms:
            word = synonym.replace(' ','_')
            local_similarity = 0
            # skip word not in the model.vocab
            if word in self.model.vocab:
                local_similarity = self.model.similarity(word,target_word)
            if local_similarity > max_similarity:
                max_similarity = local_similarity
                max_synonym = synonym              
        return max_synonym 
    
    def check_pos(self,pos):
        if pos[0] == 'V' or pos[0] == 'J':
            return True
        return False

    def get_candidates(self,lemma, _pos):
        pos_type = ''
        if _pos[0] == 'V':
            pos_type = 'v'
        if _pos[0] == 'J':
            pos_type = 'a'
        lemma = self.lemmatizer.lemmatize(lemma,pos_type)
        synsets_list = wn.synsets(lemma, pos_type)
        possible_synonyms = []
        for synset in synsets_list:
            word_list = synset.lemma_names()
            for word in word_list:
                if word == lemma or word in possible_synonyms:
                    continue
                possible_synonyms.append(word.replace('_',' '))
        return possible_synonyms

if __name__=="__main__":
    W2VMODEL_FILENAME = 'C:/Users/wu/Desktop/Columbia CS HW/NLP/HW4/hw4/hw4_files/GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    sentences = "This is a stupid movie. When I saw it in a movie theater more than half the audience left before it was half over. I stayed to the bitter end. To show fortitude? I caught it again on television and it was much funnier. Still by no means a classic, or even consistently hilarious but the family kinda grew on me. I love Jessica Lundy anyway. If you've nothing better to do and it's free on t.v. you could do worse."
    prediction = predictor.synonym_generation(sentences)
    print(sentences)
    print(prediction)
    '''
    This is a stupid movie. When I saw it in a movie theater more than half the audience
    left before it was half over. I stayed to the bitter end. To show fortitude? 
    I caught it again on television and it was much funnier. Still by no means 
    a classic, or even consistently hilarious but the family kinda grew on me. 
    I love Jessica Lundy anyway. If you've nothing better to do and it's free on t.v.
    you could do worse.

    RESULT: This is a unintelligent movie . When I see it in a movie theater more 
    than than half the audience depart before it was half over . I remain to the 
    acrimonious end . To demonstrate fortitude ? I grab it again on television 
    and it was much amusing . Still by no means a classical , or even consistently
    uproarious but the family kinda rise on me . I know Jessica Lundy anyway .
    If you 've nothing better to come and it 's complimentary on t.v .
    you could come tough . 
    '''