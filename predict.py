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

dataVocabRF = joblib.load('Files/Requirement/RF/PoSVocabRF.pkl')
JJ_NegRF, NN_NegRF, VB_NegRF, RB_NegRF, FW_NegRF = dataVocabRF['JJ_Neg'], dataVocabRF['NN_Neg'], dataVocabRF['VB_Neg'], dataVocabRF['RB_Neg'], dataVocabRF['FW_Neg']
JJ_PosRF, NN_PosRF, VB_PosRF, RB_PosRF, FW_PosRF = dataVocabRF['JJ_Pos'], dataVocabRF['NN_Pos'], dataVocabRF['VB_Pos'], dataVocabRF['RB_Pos'], dataVocabRF['FW_Pos']

dataVocabKNN = joblib.load('Files/Requirement/KNN/PoSVocabKNN.pkl')
JJ_NegKNN, NN_NegKNN, VB_NegKNN, RB_NegKNN, FW_NegKNN = dataVocabKNN['JJ_Neg'], dataVocabKNN['NN_Neg'], dataVocabKNN['VB_Neg'], dataVocabKNN['RB_Neg'], dataVocabKNN['FW_Neg']
JJ_PosKNN, NN_PosKNN, VB_PosKNN, RB_PosKNN, FW_PosKNN = dataVocabKNN['JJ_Pos'], dataVocabKNN['NN_Pos'], dataVocabKNN['VB_Pos'], dataVocabKNN['RB_Pos'], dataVocabKNN['FW_Pos']

dataVocabDT = joblib.load('Files/Requirement/DT/PoSVocabDT.pkl')
JJ_NegDT, NN_NegDT, VB_NegDT, RB_NegDT, FW_NegDT = dataVocabDT['JJ_Neg'], dataVocabDT['NN_Neg'], dataVocabDT['VB_Neg'], dataVocabDT['RB_Neg'], dataVocabDT['FW_Neg']
JJ_PosDT, NN_PosDT, VB_PosDT, RB_PosDT, FW_PosDT = dataVocabDT['JJ_Pos'], dataVocabDT['NN_Pos'], dataVocabDT['VB_Pos'], dataVocabDT['RB_Pos'], dataVocabDT['FW_Pos']

def count_tag(text, tag, listCommon, positive=True):
    words_tags = [word.split('/') for word in text.split()]
    if positive:
        return sum(1 for word, tag_ in words_tags if tag_ == tag and word in dict(listCommon))
    else:
        return sum(-1 for word, tag_ in words_tags if tag_ == tag and word in dict(listCommon))
    
def predict_text(errorId, cleanOutput):
    if errorId == 1:
        st.error("Oops! The text you entered is either too short, contains only numbers, symbols, stopwords, or is mostly made up of HTML code. Please try entering a longer and more meaningful text.")
        return
    elif errorId == 2:
        st.error("Oops! An unexpected error occurred")
        return
    else:
        modelRF = joblib.load('Files/Requirement/RF/RF-NoHuman.pkl')
        modelKNN = joblib.load('Files/Requirement/KNN/KNN-NoHuman.pkl')
        modelDT = joblib.load('Files/Requirement/DT/DT-NoHuman.pkl')
        
        scalerRF = joblib.load('Files/Requirement/RF/scalerRF.pkl')
        scalerKNN = joblib.load('Files/Requirement/KNN/scalerKNN.pkl')
        scalerDT = joblib.load('Files/Requirement/DT/scalerDT.pkl')
        
        vectorizerRF = joblib.load('Files/Requirement/RF/tfidf_vectorizerRF.pkl')
        vectorizerKNN = joblib.load('Files/Requirement/KNN/tfidf_vectorizerKNN.pkl')
        vectorizerDT = joblib.load('Files/Requirement/DT/tfidf_vectorizerDT.pkl')

        def process_data(model, scaler, vectorizer, cleanOutput, tag_dict):
            for tag, listCommon, colName in tag_dict:
                cleanOutput[colName] = cleanOutput['taggedText'].apply(lambda text: count_tag(text, tag, listCommon, positive=('Negatif' not in colName)))
            
            cleanOutput['CountTotal'] = cleanOutput.filter(like='Count').sum(axis=1)
            
            scaled_data = scaler.transform(cleanOutput[['CountTotal']])
            vectorized_data = vectorizer.transform(cleanOutput['taggedTextClean']).toarray()
            vectorized_data = vectorized_data[:, :5186]
            
            return np.hstack((vectorized_data, scaled_data))
        
        tag_columns = [
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
        
        x_RF = process_data(modelRF, scalerRF, vectorizerRF, cleanOutput.copy(), tag_columns)
        x_KNN = process_data(modelKNN, scalerKNN, vectorizerKNN, cleanOutput.copy(), tag_columns)
        x_DT = process_data(modelDT, scalerDT, vectorizerDT, cleanOutput.copy(), tag_columns)

        y_predRF = modelRF.predict(x_RF)
        y_predKNN = modelKNN.predict(x_KNN)
        y_predDT = modelDT.predict(x_DT)

        label_mapping = {1: "Positive", 0: "Negative"}
        votes = [label_mapping[y_predRF[0]], label_mapping[y_predKNN[0]], label_mapping[y_predDT[0]]]
        majority_vote = max(set(votes), key=votes.count)
        
        st.success(majority_vote)