# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import cv2
import pandas as pd
from joblib import load

# Mise en Page : en-tête. 
img = cv2.imread("pret_a_depenser.png")
st.image(img, width=200)

st.write("# Prêt à Dépenser - Dashbord interactif")
st.write("## Aide à la decision d'octroi d'un prêt à un client")

df = pd.read_csv("df_test_streamlit.csv")
#df = pd.read_csv("X_ready.csv")

client_id = st.number_input("Merci d'entrer un identifiant client :", 196534)
clients_line = df[df.SK_ID_CURR == client_id]

# s'assurer que le client fait bien parti de la bdd,
# ... et eviter un gros warning assez moche !
if len(clients_line.index) == 0:
    st.error('ATTENTION : le client ne fait pas parti de la base de donnée')
else:
    
# INFORMATIONS SUR LE CLIENT 
    st.write('#### Informations sur le client:')    

# Info à choisir/ choisies +/- par rapport aux features importances
    clients_age = int(clients_line.DAYS_BIRTH.iloc[0]/ -365)
    employed_since = round(float(clients_line["DAYS_EMPLOYED"]/ -365),2)
    anuity_revenu = round(float(clients_line['ANNUITY_INCOME_PERC']*100),2)
    Payements_rate = round(float(clients_line['PAYMENT_RATE']*100),2)
    
    show_df = pd.DataFrame({'Identifiant client': [int(client_id)],
                            'Age':clients_age,
                            'Employé depuis':str(employed_since) + ' ans',
                            "Anuité/Revenu (%)":str(anuity_revenu)+' %',
                            'Payements_rate (%)':str(Payements_rate)+' %'
                            })
    # ajouter : annuité ? durée emprunt ? 
    show_df
    st.write("**Note:**")
    st.write("'Anuité/Revenu (%)' représente la part de l'annuité dans le revenu.")
    st.write("'Payements_rate (%)' représente la part de crédit remboursé annuellement.")

# PREDICTIONS : modifier l'affichage ?
    st.write("")
    st.write(" ### Probabilité de défaut de payment du client :")
    
    clf = load('model.joblib')
    # SK_ID_CURR present dans clients_line mais pas utilisé par le modèle.
    pred_proba_client = clf.predict_proba(clients_line.iloc[:,:-1])[:, 1]
    pred_proba_client = round(pred_proba_client[0],3)
    
    if pred_proba_client >= 0.6 :
        st.write("**Forte** probabilité d'impayé.")
        st.error(pred_proba_client)
    
    elif  pred_proba_client >= 0.3 :
        st.write("probabilité d'impayé **moyenne**.")
        st.warning(pred_proba_client)
    else :
        st.write("**Faible** probabilité d'impayé.")
        st.success(pred_proba_client) 

# CLIENTS SIMILAIRES
    knn = load('knn.joblib')
    idx_neighbors = knn.kneighbors(clients_line, n_neighbors=5)
    

    #idx_neighbors    
    st.write(" #### Clients similaires :")
    L = list(idx_neighbors[1][0])
    L
    df.iloc[list(idx_neighbors[1][0]),:]
# info à afficher ... ?!

# HISTOGRAMME 


# RADARCHART
                                      
                                      
                                      
                                      
                                      
