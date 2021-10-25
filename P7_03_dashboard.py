import streamlit as st
import pandas as pd
import cv2
import plotly.graph_objects as go
from joblib import load
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler


MiseEnPage = st.container()
Info = st.container()
Prediction = st.container()
Comparaison = st.container()
Graph = st.container()

@ st.cache
def get_data(filename):
    df = pd.read_csv(filename)
    return df

def birth_transfo(x):
    birth = int(-x/365)
    return (str(birth)+ ' ans')

def employment_transfo(x):
    employment = round(-x/365,2)
    return (str(employment)+ ' ans')

def pct_transfo(x):
    pct = round(x*100,2)
    return (str(pct)+ ' %')

def sex_transfo(x):
    if x == 1:
        return 'F'
    else :
        return 'M'

X = get_data("X_ready.csv")
y = get_data("y_ready.csv")

X_y = X.copy()
X_y['TARGET'] = y['TARGET']
del X,y

with MiseEnPage:
    col1, col2 = st.columns([1,3])
    
    img = cv2.imread("pret_a_depenser.png")
    col1.image(img, width=150)

    col2.title("Prêt à Dépenser - Dashboard interactif")
    
    st.subheader(" Aide à la décision d'octroi d'un prêt à un client")
    
with Info:
    st.write("")
    st.header("Informations générales sur le client")
    
    client_id = st.number_input("Merci d'entrer un identifiant client :", 100000)
    client_idx = X_y[X_y.SK_ID_CURR == client_id].index
    client_line = X_y.iloc[client_idx, :-1]
    
    ugly_cols = ['SK_ID_CURR','DAYS_BIRTH', 'DAYS_EMPLOYED', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE', 'CODE_GENDER']
    info_client = client_line[ugly_cols]
    info_client.DAYS_BIRTH = client_line.DAYS_BIRTH.apply(birth_transfo)
    info_client.DAYS_EMPLOYED = client_line.DAYS_EMPLOYED.apply(employment_transfo)
    info_client.ANNUITY_INCOME_PERC = client_line.ANNUITY_INCOME_PERC.apply(pct_transfo)
    info_client.PAYMENT_RATE = client_line.PAYMENT_RATE.apply(pct_transfo)
    info_client.CODE_GENDER = client_line.CODE_GENDER.apply(sex_transfo)
    
    
    # s'assurer que le client fait bien parti de la bdd, et eviter un gros warning assez moche !
    if len(client_line.index) == 0:
        st.error('ATTENTION : le client ne fait pas parti de la base de donnée')
    
    else:
        nice_cols = ['Identifiant','Age', 'Employé depuis', 'Anuité/Revenu (%)', 'Payements rate (%)', 'Genre']
        client_show = go.Table(header = dict(values=nice_cols,
                                             fill_color='#AAA1CC',
                                             align = 'left',
                                             font_size= 15),
                               cells = dict(values=[info_client[cols] for cols in ugly_cols],
                                            align='center'),
                               columnwidth=[1.5, 1.3, 2, 2.2, 2.5, 1])

        fig = go.Figure(data = client_show)
        fig.update_traces(cells_height=30, cells_font_size= 15)
        fig.update_layout(margin=dict(l=0, r=0, t=15, b=10),
                          autosize=False, width=700, height=100)
        st.write(fig)
            
        st.write("**Note:**")
        st.write("*'Anuité/Revenu (%)' représente la part de l'annuité dans le revenu.*")
        st.write("*'Payements_rate (%)' représente la part de crédit remboursé annuellement.*")
    
with Prediction:
    st.write("")
    st.header("Probabilité de défaut de payment du client")
    
    if len(client_line.index) == 0:
        st.write("*Aucune information n'est disponible.*")
    

    else :
        clf = load('model.joblib')
    # SK_ID_CURR présent dans clients_line mais pas utilisé par le modèle.
        pred_proba_client = clf.predict_proba(client_line.iloc[:,:-1])[:, 1]
        pred_proba_client = round(pred_proba_client[0],3)

        col3, col4 , col4bis = st.columns([2, 1, 3])    
        if pred_proba_client >= 0.65 :
            col3.write("")
            col3.write("**Forte** probabilité d'impayé :")
            col4.error(pred_proba_client)
    
        elif  pred_proba_client >= 0.4 :
            col3.write("")
            col3.write("Probabilité d'impayé **moyenne** :")
            col4.warning(pred_proba_client)
        else :
            col3.write("")
            col3.write("**Faible** probabilité d'impayé :")
            col4.success(pred_proba_client) 
    
with Comparaison :
    st.write("")
    st.header("Clients similaires")
    
    if len(client_line.index) == 0:
        st.write("*Aucune information n'est disponible.*")
    
    else :        
        txt = "A combien de personnes voulez-vous comparer le client ?  (max.10)"
        nbr_voisins = st.number_input(txt, 1, 10)
    
        if st.button("comparer"):
            knn = load('knn.joblib')
        # le client lui même est compté comme un voisin (et même le plus proche !)
            nbr_voisins += 1
            idx_neighbors = knn.kneighbors(client_line, n_neighbors=nbr_voisins)
            L = idx_neighbors[1][0].tolist()
            
            comparison_df = pd.DataFrame()
            for indice in  L[1:]:
                new_line = pd.DataFrame.from_dict(X_y.iloc[indice,:-1].to_dict(),
                                                  orient='index').transpose()
                pred_proba_comparison = clf.predict_proba(new_line.iloc[:,:-1])[:, 1]
                new_line["PROBA"] = pred_proba_comparison
                comparison_df = comparison_df.append(new_line)
            
            ugly_cols = ['SK_ID_CURR','DAYS_BIRTH', 'DAYS_EMPLOYED', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE', 'PROBA']
            shorten_comparison = comparison_df[ugly_cols]
            shorten_comparison.DAYS_BIRTH = comparison_df.DAYS_BIRTH.apply(birth_transfo)
            shorten_comparison.DAYS_EMPLOYED = comparison_df.DAYS_EMPLOYED.apply(employment_transfo)
            shorten_comparison.ANNUITY_INCOME_PERC = comparison_df.ANNUITY_INCOME_PERC.apply(pct_transfo)
            shorten_comparison.PAYMENT_RATE = comparison_df.PAYMENT_RATE.apply(pct_transfo)
            shorten_comparison.PROBA = comparison_df.PROBA.apply(pct_transfo)
    
            nice_cols = ['Identifiant','Age', 'Employé depuis', 'Anuité/Revenu (%)', 'Payements rate (%)', "Proba. d'impayé"]
            comparison_show = go.Table(header = dict(values=nice_cols,
                                             fill_color='#AAA1CC',
                                             align = 'left',
                                             font_size= 15),
                               cells = dict(values=[shorten_comparison[cols] for cols in ugly_cols],
                                            align='center'),
                               columnwidth=[1.5, 1, 2, 2.3, 2.3, 2.3])

            fig = go.Figure(data = comparison_show)
            fig.update_traces(cells_height=30, cells_font_size= 15)
            fig.update_layout(margin=dict(l=0, r=0, t=15, b=10),
                          autosize=False, width=700, height=25+30*nbr_voisins)
            st.write(fig)
           
with Graph:
    st.write("")
    st.header("Comparaison par catégories")
    
    # Prepare radar_df : normaliser les colonnes, sauf TARGET, SK_ID_CURR et GENDER_CODE
    categories = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ANNUITY_INCOME_PERC', 'AMT_ANNUITY', 'PAYMENT_RATE']
    closing_categories = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ANNUITY_INCOME_PERC', 'AMT_ANNUITY', 'PAYMENT_RATE', 'DAYS_BIRTH']
        
    scaler = MinMaxScaler(feature_range=(0,1))
        
    extract_df = X_y[categories]
    
    extract_df['DAYS_BIRTH'] = extract_df['DAYS_BIRTH'].apply(lambda x : -x/365)
    extract_df['DAYS_EMPLOYED'] = extract_df['DAYS_EMPLOYED'].apply(lambda x : -x/365)
    extract_array = scaler.fit_transform(extract_df)
    radar_df = pd.DataFrame(extract_array)
    radar_df.columns=categories
    radar_df['TARGET'] = X_y['TARGET']
    radar_df['SK_ID_CURR'] = X_y['SK_ID_CURR']
    
    # ------------------------------ COMPARAISON PAR GENRE -------------------------------
    st.write("")
    col5, col6 = st.columns(2)
    col5.write("##### Proportion d'impayés par genre")
              
    # HISTOGRAM
    gender_histo = X_y.groupby(['CODE_GENDER', 'TARGET']).size().reset_index()
    gender_histo['CODE_GENDER'] = gender_histo['CODE_GENDER'].apply(lambda x :'F' if x==1 else 'M')
    gender_histo['TARGET'] = gender_histo['TARGET'].apply(lambda x :int(x))
    gender_histo['Percentage'] = X_y.groupby(
        ['CODE_GENDER', 'TARGET']).size().groupby(level=0).apply(lambda x: 100 * x/float(x.sum())).values
    gender_histo.columns = ['CODE_GENDER', 'TARGET', 'COUNTS', 'Percentage']
    gender_histo['Percentage'] = gender_histo['Percentage'].map('{:,.2f}%'.format)
        
    fig = px.bar(gender_histo, x = 'CODE_GENDER', y = 'COUNTS', color ='TARGET', barmode ='stack', 
                 text=gender_histo['Percentage'])
    fig.update_layout(
                      xaxis_title = 'Genre', xaxis_title_font_size=18,
                      yaxis_title = 'Effectifs', yaxis_title_font_size=18,
                      margin=dict(l=0, r=0, t=20, b=10),
                      autosize=False, width = 300, height = 350,
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h"))
    fig.update_traces(marker=dict(color=["Chartreuse","DeepPink", "Chartreuse",  "DeepPink"]))

    col5.write(fig)

    # RADARCHART
    col6.write("##### Comparaison aux clients du même sexe")
    
    if len(client_line.index) == 0:
        col6.write("")
        col6.write("")
        col6.write("")
        col6.write("")
        col6.write("")
        col6.write("")
        col6.write("*Aucune information n'est disponible.*")
    
    else :

        col6.write('sexe du client :  **'+ sex_transfo(client_line.CODE_GENDER.iloc[0])+'**')
        
        radar_df['CODE_GENDER'] = X_y['CODE_GENDER']
        
        radar_gender = radar_df[radar_df.CODE_GENDER == client_line.CODE_GENDER.iloc[0]]

        sub_gender0 = radar_gender[radar_gender['TARGET']==0]
        sub_gender1 = radar_gender[radar_gender['TARGET']==1]
        sub_genderClient = radar_gender[categories][radar_gender.SK_ID_CURR == client_id]
        
        dataGender0= [np.mean(sub_gender0[col]) for col in closing_categories]
        dataGender1= [np.mean(sub_gender1[col]) for col in closing_categories]
        dataGenderClient= [np.mean(sub_genderClient[col]) for col in closing_categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=dataGender0, theta=closing_categories, name='crédit remboursé'))
        fig.add_trace(go.Scatterpolar(r=dataGender1, theta=closing_categories, name='Impayé'))
        fig.add_trace(go.Scatterpolar(r=dataGenderClient, theta=closing_categories,
                                      fill='toself', name='client'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=True,
                          margin=dict(l=0, r=0, t=30, b=10),
                          autosize=False, width = 650, height = 320)
        col6.write(fig)
    
    
    # ------------------------------ COMPARAISON PAR SECTEUR -------------------------------
    st.write("")

    st.write("##### Proportion d'impayés par type d'emploi")    
    #prepare df
    X_ORGANIZATION_TYPE = get_data('X_ORGANIZATION_TYPE.csv')
    X_y = X_y.merge(X_ORGANIZATION_TYPE, how='inner', on='SK_ID_CURR')
    del X_ORGANIZATION_TYPE
    
    # HISTOGRAM
    sector_histo = X_y.groupby(['ORGANIZATION_TYPE', 'TARGET']).size().reset_index()
    sector_histo['Percentage'] = X_y.groupby(
        ['ORGANIZATION_TYPE', 'TARGET']).size().groupby(level=0).apply(lambda x: 100 * x/float(x.sum())).values
    sector_histo.columns = ['ORGANIZATION_TYPE', 'TARGET', 'COUNTS', 'Percentage']
    sector_histo['Percentage'] = sector_histo['Percentage'].map('{:,.2f}%'.format)
    #sector_histo

    fig = px.bar(sector_histo, x = 'ORGANIZATION_TYPE', y = 'COUNTS', color ='TARGET', text=sector_histo['Percentage'])
    fig.update_traces(marker=dict(color=["Chartreuse","DeepPink", "Chartreuse","DeepPink", "Chartreuse","DeepPink",
                                         "Chartreuse","DeepPink", "Chartreuse","DeepPink", "Chartreuse","DeepPink",
                                         "Chartreuse","DeepPink", "Chartreuse","DeepPink", "Chartreuse","DeepPink",
                                         "Chartreuse","DeepPink", "Chartreuse","DeepPink", "Chartreuse","DeepPink",
                                         "Chartreuse","DeepPink", "Chartreuse","DeepPink", "Chartreuse","DeepPink",
                                         "Chartreuse","DeepPink", "Chartreuse","DeepPink", "Chartreuse","DeepPink"]))
    fig.update_layout(yaxis_title = "Effectif ", yaxis_title_font_size=15,
                      margin=dict(l=0, r=0, t=20, b=10),
                      autosize=False, width = 800, height = 500)
   
    st.write(fig)
    
    # RADARCHART
    st.write("##### Comparaison avec les clients travaillant dans le même secteur")
    
    if len(client_line.index) == 0:
        st.write("*Aucune information n'est disponible.*")
    
    else :
        radar_df['ORGANIZATION_TYPE'] = X_y['ORGANIZATION_TYPE']
        sector_client_line = X_y.iloc[client_idx, :]
        
        st.write("Secteur d'activité du client : " + '**'+ sector_client_line.ORGANIZATION_TYPE.iloc[0]+'**')
        
        radar_sector = radar_df[radar_df.ORGANIZATION_TYPE == sector_client_line.ORGANIZATION_TYPE.iloc[0]]

        sub_sector0 = radar_sector[radar_sector['TARGET']==0]
        sub_sector1 = radar_sector[radar_sector['TARGET']==1]
        sub_sectorClient = radar_sector[categories][radar_sector.SK_ID_CURR == client_id]
       
        dataSector0= [np.mean(sub_sector0[col]) for col in closing_categories]
        dataSector1= [np.mean(sub_sector1[col]) for col in closing_categories]
        dataSectorClient= [np.mean(sub_sectorClient[col]) for col in closing_categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=dataSector0, theta=closing_categories, name='crédit remboursé'))
        fig.add_trace(go.Scatterpolar(r=dataSector1, theta=closing_categories, name='Impayé'))
        fig.add_trace(go.Scatterpolar(r=dataSectorClient, theta=closing_categories,
                                      fill='toself', name='client'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=True,
                          margin=dict(l=0, r=0, t=30, b=10),
                          autosize=False, width = 650, height = 320)
        st.write(fig)