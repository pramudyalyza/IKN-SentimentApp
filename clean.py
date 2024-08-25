import re
import sys
import html
import nltk
import json
import spacy
import pickle
import string
import warnings
import numpy as np
import unicodedata
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from indoNLP.preprocessing import replace_slang
from indoNLP.preprocessing import emoji_to_words
from indoNLP.preprocessing import replace_word_elongation
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')

with open('Files/Requirement/typoList.json', 'r') as file:
    typoList = json.load(file)

with open('Files/Requirement/posTagger.pickle', 'rb') as file:
    POStagger = pickle.load(file)

def remove_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def convert_html_unicode(text):
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'["&.]', ' ', text)
    return text

def convert_emoji(text):
    res = emoji_to_words(text)
    res = re.sub(r'[^\w\s]|_+', ' ', res)
    return res

def replace_elongation(text):
    return replace_word_elongation(text)

def replace_slangWord(text):
    replaced_text = replace_slang(text)
    replaced_text = replaced_text.replace('enggak', 'tidak')
    return replaced_text

def replace_typos(text):
    kata_kata = text.split()
    for i, kata in enumerate(kata_kata):
        if kata.lower() in typoList:
            kata_kata[i] = typoList[kata.lower()]
    clean_text = ' '.join(kata_kata)
    return clean_text

def tagging(text):
  res = POStagger.tag(word_tokenize(text))
  return ' '.join([f'{word}/{tag}' for word, tag in res])

def remove_stopwords(text):
    stopwords_indo = set(stopwords.words('indonesian'))
    tokens = text.split()
    filtered_tokens = []

    for token in tokens:
        word, pos_tag = token.split('/')
        if word.lower() not in stopwords_indo or pos_tag == 'NEG':
            filtered_tokens.append(token)

    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_pos_tags(text):
    words = text.split()
    cleaned_words = [word.split('/')[0] for word in words]
    return ' '.join(cleaned_words)

def process_text(text):
    df = pd.DataFrame({'textDisplay': [text]})
    try:
        df['textNoMentions'] = df['textDisplay'].apply(lambda x: re.sub(r'@+\S*', '', x))
        df['textNoUrl'] = df['textNoMentions'].apply(lambda x: re.sub(r'<a href\S+<\/a>', '', x))
        df['textNoHTML'] = df['textNoUrl'].apply(remove_html)
        df['textNoNumbers'] = df['textNoHTML'].apply(lambda x: re.sub(r'\d+', '', x))
        df['textNoPunct'] = df['textNoNumbers'].apply(lambda x: re.sub(r'[.,;:!?/\\,#@$()%"℅]', ' ', x))
        df['textNoExtraSpaces'] = df['textNoPunct'].apply(lambda x: ' '.join(x.split()))
        df['textLower'] = df['textNoExtraSpaces'].str.lower()
        df['textConvertHTML'] = df['textLower'].apply(lambda x: convert_html_unicode(x))
        df['textConvertEmoji'] = df['textConvertHTML'].apply(convert_emoji)
        df['textNoElongation'] = df['textConvertEmoji'].apply(replace_elongation)
        df['textNoSlang'] = df['textNoElongation'].apply(replace_slangWord)
        df['textNoTypo'] = df['textNoSlang'].apply(replace_typos)
        df['textTagging'] = df['textNoTypo'].apply(tagging)
        df['taggedStop'] = df['textTagging'].apply(remove_stopwords)
        df['taggedStopClean'] = df['taggedStop'].apply(remove_pos_tags)

        if df['taggedStopClean'].iloc[0] == '':
            return 1, None, None
    
    except Exception as e:
        return 2, None, e

    res = df[['textDisplay', 'taggedStopClean', 'taggedStop']]
    res = res.rename(columns = {'textDisplay':'fullText', 'taggedStopClean':'taggedTextClean', 'taggedStop':'taggedText'})

    return 0, res, None
