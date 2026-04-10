import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
import re
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.
    qwerty_typos = {
         'q': 'was', 'w': 'qeasd', 'e': 'wrsdf', 'r': 'etdfg', 't': 'ryfgh', 'y': 'tughj',
         'u': 'yihjk',
         'i': 'uojkl',
         'o': 'ipkl',
         'p': 'ol',
         'a': 'qwszx',
         's': 'qweadzxc', 'd': 'wersfxcv', 'f': 'ertdgcvb', 'g': 'rtyfhvbn', 'h': 'tyugjbnm', 'j': 'yuihknm,',
         'k': 'uiojlm,.', 'l': 'iopk,.', 'z': 'asx', 'x': 'asdzc', 'c': 'sdfxv', 'v': 'dfgcb', 'b': 'fghvn', 
         'n': 'ghjbm', 'm': 'hjkn,'}
    
    # protected_words = {'not', 'no', 'never', 'none', 'cannot', "cant", "wont", "dont", "didnt", "isnt", "wasnt"}

    def get_synonym(word):
        synsets = wordnet.synsets(word)
        if not synsets:
            return word
        synonyms = []
        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    synonyms.append(name)
        
        if not synonyms:
            return word
        
        # Return a random synonym from the list
        return random.choice(list(set(synonyms)))

    def add_typos(token):
        candidate_positions = [i for i, ch in enumerate(token.lower()) if ch in qwerty_typos]
        if not candidate_positions:
            return token
        pos = random.choice(candidate_positions)
        original_char = token[pos]
        replacement = random.choice(qwerty_typos[original_char.lower()])
        if original_char.isupper():
            replacement = replacement.upper()
        chars = list(token)
        chars[pos] = replacement
        return ''.join(chars)

    tokens = re.findall(r"\w+|[^\w\s]", example["text"], flags=re.UNICODE)
    transformed_tokens = []
    
    # Probabilities for transformations
    typo_prob = 0.15
    synonym_prob = 0.15

    for tok in tokens:
        if tok.isalpha():
            r = random.random()
            if r < typo_prob:
                transformed_tokens.append(add_typos(tok))
            elif r < (typo_prob + synonym_prob):
                transformed_tokens.append(get_synonym(tok))
            else:
                transformed_tokens.append(tok)
        else:
            transformed_tokens.append(tok)

    example["text"] = TreebankWordDetokenizer().detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######
    return example
