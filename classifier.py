from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools

from nltk.corpus import stopwords
from nltk.util import ngrams
import textstat, spacy, nltk
import pandas as pd, os
import numpy as np
import string, re
import random

nltk.download('stopwords')
stopwords = stopwords.words('english')
genders = ['Male', 'Female']
np.random.seed(0)


class JD_Purifier():
    def __init__(self):
        self.lookups = {
            "masculine": ['active', 'adventurous', 'aggress', 'ambitio', 'analy', 'assert', 'athlet', 'autonom',
                          'battle', 'boast', 'challeng', 'champion', 'compet', 'confident', 'courag', 'decid',
                          'decision', 'decisive', 'defend', 'determin', 'domina', 'dominant', 'driven', 'fearless',
                          'fight', 'force', 'greedy', 'head strong', 'headstrong', 'hierarch', 'hostil', 'impulsive',
                          'independen', 'individual', 'intellect', 'lead', 'logic', 'objective', 'opinion', 'outspoken',
                          'persist', 'principle', 'reckless', 'self confiden', 'self relian', 'self sufficien',
                          'selfconfiden', 'selfrelian', 'selfsufficien', 'stubborn', 'superior', 'unreasonab',
                          'capable', 'certain', 'focus', 'benefit', 'trust ', 'trusting', 'acceptance ', 'accepting',
                          'appreciative ', 'appreciation', 'admire ', 'admiration', 'approval', 'encouragement',
                          'power', 'strenght', 'competency ', 'competence', 'efficient ', 'efficiency', 'achievement',
                          'honor', 'pride', 'dignity', 'solution ', 'solutions', 'success', 'skills', 'autonomy',
                          'love', 'serve', 'support', 'give ', 'giving', 'provide', 'devoted', 'fulfill', 'caretaker',
                          'space', 'useful', 'rational', 'strategy ', 'strategic', 'plan ', 'planning', 'analytic ',
                          'analytical', 'reasonable', 'consider', 'analyse ', 'analysing', 'believe', 'opinion',
                          'suggestion', 'think', 'prove themselves', 'achieve results', 'feel good about himself',
                          'doing things by himself', 'loving acceptance', 'feeling needed', 'someone to serve',
                          'good enough', 'fulfill others', 'silent acceptance', 'comforting love', 'common sense',
                          'point of view', 'active', 'adventurous', 'aggress', 'ambitio', 'analy', 'assert', 'athlet',
                          'autonom', 'boast', 'challeng', 'compet', 'confident', 'courag', 'decide', 'decisive',
                          'decision', 'determin', 'dominant', 'domina', 'force', 'greedy', 'headstrong', 'hierarch',
                          'hostil', 'implusive', 'independen', 'individual', 'intellect', 'lead', 'logic', 'masculine',
                          'objective', 'opinion', 'outspoken', 'persist', 'principle', 'reckless', 'stubborn',
                          'superior', 'self confiden', 'self sufficien', 'self relian'],
            "feminine": ['agree', 'affectionate', 'child', 'cheer', 'collab', 'commit', 'communal', 'compassion',
                         'connect', 'considerate', 'cooperat', 'co operat', 'depend', 'emotiona', 'empath', 'feel',
                         'flatterable', 'gentle', 'honest', 'interdependen', 'interpersona', 'inter personal',
                         'inter dependen', 'inter persona', 'kind', 'kinship', 'loyal', 'modesty', 'nag', 'nurtur',
                         'pleasant', 'polite', 'quiet', 'respon', 'sensitiv', 'submissive', 'support', 'sympath',
                         'tender', 'together', 'trust', 'understand', 'warm', 'whin', 'enthusias', 'inclusive', 'yield',
                         'share', 'sharin', 'affectionate', 'child', 'cheer', 'commit', 'communal', 'compassion',
                         'connect', 'considerate', 'cooperat', 'depend', 'emotiona', 'empath', 'feminine',
                         'flatterable', 'gentle', 'honest', 'interdependen', 'interpersona', 'kindkinship', 'loyal',
                         'modesty', 'nag', 'nurtur', 'pleasant', 'polite', 'quiet', 'respon', 'sensitiv', 'submissive',
                         'support', 'sympath', 'tender', 'together', 'trust', 'understand', 'warm', 'whin', 'yield',
                         'ease', 'permission', 'kindness', 'appreciation', 'caring', 'respect', 'devotion',
                         'validation', 'reassurance', 'respectful', 'love', 'communication', 'beauty', 'relationships',
                         'helping', 'sharing', 'relating', 'harmony', 'community', 'talking', 'intimate', 'life',
                         'healing', 'growth', 'intuitive', 'companionship', 'receive ', 'receiving', 'cherished',
                         'creativity', 'reassurance', 'worthy', 'supported ', 'supporting', 'nurture ', 'nurturing ',
                         'nurtured', 'feel', 'emotion'],
            "superlatives": ['expert', 'perfection', 'rockstar', 'specialist', 'authority', 'pundit', 'oracle',
                             'resource person', 'adept', 'maestro', 'virtuoso', 'master', 'past master', 'professional',
                             'genius', 'wizard', 'connoisseur', 'aficionado', 'cognoscenti', 'cognoscente', 'doyen',
                             'savant', 'ace', 'buff', 'ninja', 'pro', 'whizz', 'hotshot', 'old hand', 'alpha geek',
                             'dab hand', 'maven', 'crackerjack'],
            "relationships": ['family', 'child', 'parent', 'women', 'mother', 'father', 'son', 'daughter', 'kids',
                              'kid', 'married', 'household', 'home', 'sibling'],
            "strict_words": ['disqualified', 'rejected', 'must', 'should', 'required', 'banned', 'barred', 'disbarred',
                             'debarred', 'eliminated', 'precluded', 'disentitled', 'ineligible', 'unfit', 'unqualified',
                             'essential', 'desirable', 'desired'],
            "strict_phrases": ['only be', 'who fail', 'not allowed', 'should have', 'is required', 'subject to',
                               'will not be considered', 'cannot be appointed', 'who lack'],
            "mas_pronouns": ["he", "his", "him", "himself"],
            "fem_pronouns": ["she", "her", "herself"],
        }

    ## function to clean a text
    def _cleanup(self, text):
        text = text.lower()
        text = " ".join([c for c in text.split() if c not in stopwords])
        for c in string.punctuation:
            text = text.replace(c, " ")
        text = " ".join([c for c in text.split() if c not in stopwords])

        words = []
        ignorewords = []
        for wrd in text.split():
            if len(wrd) <= 2:
                continue
            if wrd in ignorewords:
                continue
            words.append(wrd)
        text = " ".join(words)
        return text

    def _check_presence(self, flag, txt, condition):
        matched = {}
        txt = " " + txt.lower() + " "
        for wrd in self.lookups[flag]:
            if condition == "exact":
                cnt = txt.count(" " + wrd.lower() + " ")
                if cnt > 0:
                    matched[wrd.lower()] = cnt
            elif condition == "startswith":
                cnt = txt.count(" " + wrd.lower())
                if cnt > 0:
                    matched[wrd.lower()] = cnt

        return matched


import numpy as np


def _gender_bias(txt, filename):
    ## create the object
    jdp = JD_Purifier()
    cln = jdp._cleanup(txt)

    ## word choice
    mas_words = jdp._check_presence("masculine", cln, condition='startswith')
    fem_words = jdp._check_presence("feminine", cln, condition='startswith')
    mas_wc, fem_wc = sum(mas_words.values()), sum(fem_words.values())

    ## pronoun usage
    mas_prns = jdp._check_presence("mas_pronouns", cln, condition='exact')
    fem_prns = jdp._check_presence("fem_pronouns", cln, condition='exact')
    masp_wc, femp_wc = sum(mas_prns.values()), sum(fem_prns.values())

    ## relationships
    relationships = jdp._check_presence("relationships", cln, condition='exact')
    relationships_wc = sum(relationships.values())

    ## superlatives useage
    superlatives = jdp._check_presence("superlatives", cln, condition='exact')
    superlatives_wc = len(superlatives.values())


    doc = {"mas_wc": mas_wc, "fem_wc": fem_wc,
           "masp_wc": masp_wc, "femp_wc": femp_wc,
           "relationships_wc": relationships_wc,
           "superlatives_wc": superlatives_wc,
           "superlatives_wrds": superlatives,
           "mas_words": mas_words,
           "fem_words": fem_words}
    return doc

def get_gender_dict(df):
    femdict, masdict = {}, {}
    for d in df['fem_words']:
        for k, v in d.items():
            if k not in femdict:
                femdict[k] = v
            femdict[k] += v
    for d in df['mas_words']:
        for k, v in d.items():
            if k not in masdict:
                masdict[k] = v
            masdict[k] += v
    return femdict,masdict


def find_score(x):
    d = x['difference'] * -1
    if d > 50:
        f = 10
    elif d > 42:
        f = 8.2
    elif d > 33:
        f = 6.5
    elif d > 24:
        f = 5.4
    elif d > 15:
        f = 3.8
    elif d > 10:
        f = 3.0
    elif d > 5:
        f = 1.5
    elif d > 1:
        f = 1
    else:
        f = 0

    d = x['superlatives_wc']
    if d > 5:
        f2 = 10
    elif d > 3:
        f2 = 7
    elif d > 1:
        f2 = 4
    else:
        f2 = 0

    d = x['relationships_wc']
    if d < 5:
        f3 = 10
    elif d < 3:
        f3 = 7
    elif d < 1:
        f3 = 4
    else:
        f3 = 0

    score = f * 0.70 + f2 * 0.20 + f3 * 0.10
    return score