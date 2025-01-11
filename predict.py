import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')

RFVocab = joblib.load('Files/Requirement/RF/RFVocab.pkl')
JJ_NegRF, NN_NegRF, VB_NegRF, RB_NegRF, FW_NegRF = RFVocab['JJ_Neg'], RFVocab['NN_Neg'], RFVocab['VB_Neg'], RFVocab['RB_Neg'], RFVocab['FW_Neg']
JJ_PosRF, NN_PosRF, VB_PosRF, RB_PosRF, FW_PosRF = RFVocab['JJ_Pos'], RFVocab['NN_Pos'], RFVocab['VB_Pos'], RFVocab['RB_Pos'], RFVocab['FW_Pos']

SVMVocab = joblib.load('Files/Requirement/SVM/SVMVocab.pkl')
JJ_NegSVM, NN_NegSVM, VB_NegSVM, RB_NegSVM, FW_NegSVM = SVMVocab['JJ_Neg'], SVMVocab['NN_Neg'], SVMVocab['VB_Neg'], SVMVocab['RB_Neg'], SVMVocab['FW_Neg']
JJ_PosSVM, NN_PosSVM, VB_PosSVM, RB_PosSVM, FW_PosSVM = SVMVocab['JJ_Pos'], SVMVocab['NN_Pos'], SVMVocab['VB_Pos'], SVMVocab['RB_Pos'], SVMVocab['FW_Pos']

LRVocab = joblib.load('Files/Requirement/LR/LRVocab.pkl')
JJ_NegLR, NN_NegLR, VB_NegLR, RB_NegLR, FW_NegLR = LRVocab['JJ_Neg'], LRVocab['NN_Neg'], LRVocab['VB_Neg'], LRVocab['RB_Neg'], LRVocab['FW_Neg']
JJ_PosLR, NN_PosLR, VB_PosLR, RB_PosLR, FW_PosLR = LRVocab['JJ_Pos'], LRVocab['NN_Pos'], LRVocab['VB_Pos'], LRVocab['RB_Pos'], LRVocab['FW_Pos']

def count_tag(text, tag, listCommon, positive=True):
    words_tags = [word.split('/') for word in text.split()]
    if positive:
        return sum(1 for word, tag_ in words_tags if tag_ == tag and word in dict(listCommon))
    else:
        return sum(-1 for word, tag_ in words_tags if tag_ == tag and word in dict(listCommon))
    
def predict_text(errorId, cleanOutput, errorMessage):
    if errorId == 1:
        st.error("Oops! The text you entered is either too short, contains only numbers, symbols, stopwords, or is mostly made up of HTML code. Please try entering a longer and more meaningful text.")
        return
    elif errorId == 2:
        st.error(errorMessage)
        st.error("Oops! An unexpected error occurred")
        return
    else:
        modelRF = joblib.load('Files/Requirement/RF/RFModel.pkl')
        modelSVM = joblib.load('Files/Requirement/SVM/SVMModel.pkl')
        modelLR = joblib.load('Files/Requirement/LR/LRModel.pkl')
        
        scalerRF = joblib.load('Files/Requirement/RF/RFScaler.pkl')
        scalerSVM = joblib.load('Files/Requirement/SVM/SVMScaler.pkl')
        scalerLR = joblib.load('Files/Requirement/LR/LRScaler.pkl')
        
        vectorizerRF = joblib.load('Files/Requirement/RF/RFVectorizer.pkl')
        vectorizerSVM = joblib.load('Files/Requirement/SVM/SVMVectorizer.pkl')
        vectorizerLR = joblib.load('Files/Requirement/LR/LRVectorizer.pkl')

        def process_data(model, scaler, vectorizer, cleanOutput, tag_dict):
            for tag, listCommon, colName in tag_dict:
                cleanOutput[colName] = cleanOutput['taggedText'].apply(lambda text: count_tag(text, tag, listCommon, positive=('Negatif' not in colName)))
            
            cleanOutput['CountTotal'] = cleanOutput.filter(like='Count').sum(axis=1)
            
            scaled_data = scaler.transform(cleanOutput[['CountTotal']])
            vectorized_data = vectorizer.transform(cleanOutput['taggedTextClean']).toarray()
            vectorized_data = vectorized_data[:, :2467]
            
            return np.hstack((vectorized_data, scaled_data))
        
        tag_columnsLR = [
            ('JJ', JJ_NegLR, 'CountJJ_Negatif'),
            ('NN', NN_NegLR, 'CountNN_Negatif'),
            ('VB', VB_NegLR, 'CountVB_Negatif'),
            ('RB', RB_NegLR, 'CountRB_Negatif'),
            ('FW', FW_NegLR, 'CountFW_Negatif'),
            ('JJ', JJ_PosLR, 'CountJJ_Positif'),
            ('NN', NN_PosLR, 'CountNN_Positif'),
            ('VB', VB_PosLR, 'CountVB_Positif'),
            ('RB', RB_PosLR, 'CountRB_Positif'),
            ('FW', FW_PosLR, 'CountFW_Positif')]
        
        tag_columnsSVM = [
            ('JJ', JJ_NegSVM, 'CountJJ_Negatif'),
            ('NN', NN_NegSVM, 'CountNN_Negatif'),
            ('VB', VB_NegSVM, 'CountVB_Negatif'),
            ('RB', RB_NegSVM, 'CountRB_Negatif'),
            ('FW', FW_NegSVM, 'CountFW_Negatif'),
            ('JJ', JJ_PosSVM, 'CountJJ_Positif'),
            ('NN', NN_PosSVM, 'CountNN_Positif'),
            ('VB', VB_PosSVM, 'CountVB_Positif'),
            ('RB', RB_PosSVM, 'CountRB_Positif'),
            ('FW', FW_PosSVM, 'CountFW_Positif')]
        
        tag_columnsRF = [
            ('JJ', JJ_NegRF, 'CountJJ_Negatif'),
            ('NN', NN_NegRF, 'CountNN_Negatif'),
            ('VB', VB_NegRF, 'CountVB_Negatif'),
            ('RB', RB_NegRF, 'CountRB_Negatif'),
            ('FW', FW_NegRF, 'CountFW_Negatif'),
            ('JJ', JJ_PosRF, 'CountJJ_Positif'),
            ('NN', NN_PosRF, 'CountNN_Positif'),
            ('VB', VB_PosRF, 'CountVB_Positif'),
            ('RB', RB_PosRF, 'CountRB_Positif'),
            ('FW', FW_PosRF, 'CountFW_Positif')]
        
        x_LR = process_data(modelLR, scalerLR, vectorizerLR, cleanOutput.copy(), tag_columnsLR)
        x_SVM = process_data(modelSVM, scalerSVM, vectorizerSVM, cleanOutput.copy(), tag_columnsSVM)
        x_RF = process_data(modelRF, scalerRF, vectorizerRF, cleanOutput.copy(), tag_columnsRF)

        y_predLR = modelLR.predict(x_LR)
        y_predSVM = modelSVM.predict(x_SVM)
        y_predRF = modelRF.predict(x_RF)

        label_mapping = {1: "Positive", 0: "Negative"}
        votes = [label_mapping[y_predLR[0]], label_mapping[y_predSVM[0]], label_mapping[y_predRF[0]]]
        majority_vote = max(set(votes), key=votes.count)
        
        st.success(majority_vote)
