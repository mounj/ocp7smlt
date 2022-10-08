# Chargement des librairies nécessaires
from cProfile import label
from operator import contains
from unittest import result
import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
import numpy as np
import matplotlib
#import seaborn as sns
import requests
import json
import pickle
import os
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Bandeau d'accueil
st.sidebar.title("Prêt à dépenser")
image = Image.open("img_depense.jpg")
st.sidebar.image(image)

# Paramètre d'activation de l'usage de l'API en ligne ou non
LRSMOTE_URI = 'https://ocp7gitapi.herokuapp.com/predict'
with_API = True
online_input = st.sidebar.radio('Connexion API:', ('Oui', 'Non'))
if online_input == 'Oui':
    with_API = False
else :
    with_API = True

# Chargement du modèle - Possibilité de choisir un modèle au besoin
current_path = os.getcwd()
model_name = 'LR_SMOTE.pkl'
credit_path = os.path.join(current_path, model_name)
with open(credit_path, 'rb') as handle:
    model = pickle.load(handle)

# Méthode de prédiction en local ou en ligne
def prediction(X):
    '''Réalise une prédiction à partir du modèle en local'''
    prediction = model.predict(X)
    return prediction


def request_prediction(model_uri, data):
    '''Réalise une prédiction à partur de l'API hébergée sur Heroku'''
    response = {}
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data.to_json()}

    response = requests.request(method='POST',
                                headers=headers,
                                url=model_uri,
                                json=data_json)

    if response.status_code != 200:
        raise Exception("Request failed with status {}, {}".format(
            response.status_code, response.text))

    return response


def feature_importance(df):
    feature_names = df.drop(columns=['TARGET','PREDICTED', 'result']).columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(df.drop(columns=['TARGET', 'PREDICTED', 'result']),
               df['TARGET'])

    forest_importances = pd.Series(
        forest.feature_importances_,
        index=feature_names).sort_values(ascending=False)

    return forest_importances
# pd.DataFrame(forest_importances, columns=['importance']).reset_index().rename(columns={'index': 'feature'})


# Affichage des features importantes
def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(),
                    labels={
                        'x': 'Importance Value',
                        'index': 'Feature'
                    },
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=1000,
                    height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)

# Chargement du fichier de démonstration
def chargement_data(path):
    dataframe = pd.read_csv(path)
    liste_id = dataframe['SK_ID_CURR'].astype(int).tolist()
    liste_id.insert(0,0)
    return dataframe, liste_id

# Utilisation des données clientes pour présenter l'objet de l'étude d'un prêt
if with_API:
    examples_file = 'new_train_cleaned_application.csv'  #'application_API.csv'
else :
    examples_file = 'new_train_cleaned_application.csv'
dataframe, liste_id = chargement_data(examples_file)

def main_page():
    """ Calcul de la prédiction pour le client issue de la base"""
    st.title('Calcul du risque de remboursement de prêt')
    st.subheader('Prédictions de scoring client')

    if 'client' not in st.session_state:
        st.session_state.client = 0
    else:
        id_input = st.session_state.client

    id_input = st.selectbox('Choisissez le client que vous souhaitez visualiser', liste_id)
    st.session_state.client = id_input

    client_infos = dataframe[dataframe['SK_ID_CURR'] == id_input].drop(
        ['SK_ID_CURR'], axis=1)
    client_infos.to_dict(orient='records')

    result = ""

    if id_input == 0:
        st.write('Please select a client')
    else:
        X1 = dataframe[dataframe['SK_ID_CURR'] == id_input]
        # X = X1[[
        #     'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
        #     'PAYMENT_RATE', 'DAYS_REGISTRATION', 'INCOME_CREDIT_PERC',
        #     'ANNUITY_INCOME_PERC', 'AMT_ANNUITY', 'DAYS_LAST_PHONE_CHANGE',
        #     'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'INCOME_PER_PERSON',
        #     'DAYS_EMPLOYED_PERC', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE',
        #     'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
        #     'AMT_REQ_CREDIT_BUREAU_YEAR'
        # ]]
        X = X1 [[
               'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
          'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
          'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'PAYMENT_RATE',
          'FLAG_PHONE']]

        if with_API:
            st.write('local model')
            result = prediction(X)
        else:
            st.write('API model')
            result = int(json.loads(request_prediction(LRSMOTE_URI, X).content)["prediction"])

        if result == 1:
            if int(X1['TARGET']) == 1:
                pred = 'rejeté (True Positive)'
            else:
                pred = 'approuvé (False Positive)'
        else:
            if int(X1['TARGET']) == 1:
                pred = 'rejeté (False Negative)'
            else:
                pred = 'approuvé (True Negative)'

        if "approuvé" in pred:
            st.success('Votre crédit est {}'.format(pred))#, icon="✅")
        else:
            st.error('votre crédit est {}'.format(pred)) #, icon="🚨")

def page2():
    st.title("Interprétabilité du modèle")

    #st.write ('--- session_state.client page 2')
    id_input = st.session_state.client

    st.write('Pour le client  ', id_input,
             ' poids des variables dans le modèle rfc')

    # informations du client
    st.header("Informations du client")
    examples_file = 'application.csv'
    # examples_file = 'new_train_cleaned_application.csv'  #'application.csv'
    application, liste_id = chargement_data(examples_file)
    # application.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_infos_client = application[application['SK_ID_CURR'] == id_input]
    st.write(X_infos_client)

    # st.header("FLAG_PHONE / EXT_SOURCE_3 / target")
    # fig = px.bar(application,
    #              x="FLAG_PHONE",
    #              y="EXT_SOURCE_3",
    #              color="TARGET",)
    # #  notched=True)
    # st.plotly_chart(fig)
    dataframe['PREDICTED'] = pd.DataFrame(
        prediction(dataframe[[
            'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
            'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'PAYMENT_RATE',
            'FLAG_PHONE'
        ]]))

    dataframe['result'] = (dataframe['TARGET'] * 2 - dataframe['PREDICTED'])

    cat_map = {-1: "FP", 0: "TN", 1: "TP", 2: "FN"}
    dataframe['result'] = dataframe['result'].map(cat_map)
    # # comparatif = application.map(lambda x : [])
    # dataframe['CLASSIFICATION'] = dataframe.applymap(result_prediction('TARGET','PREDICTED'))

    # st.write(dataframe[dataframe['SK_ID_CURR'] == id_input][[
    #     'SK_ID_CURR',
    #     'TARGET',
    #     'PREDICTED',
    #     'result',
    #     'EXT_SOURCE_3',
    #     'EXT_SOURCE_2',
    #     'EXT_SOURCE_1',
    #     'FLAG_PHONE',
    #     'OBS_60_CNT_SOCIAL_CIRCLE',
    #     'OBS_30_CNT_SOCIAL_CIRCLE',
    #     'AMT_REQ_CREDIT_BUREAU_YEAR',
    #     'CNT_CHILDREN',
    #     'CNT_FAM_MEMBERS',
    #     'PAYMENT_RATE'
    # ]])

    # st.write(dataframe.describe())
    # fig = px.bar(dataframe,
    #              x="result",
    #              y="result",)
    # st.plotly_chart(fig)

    # fig = px.box(dataframe,
    #              x="result",
    #              y="EXT_SOURCE_2",
    #              color="TARGET",
    #              notched=True)
    # st.plotly_chart(fig)

    # fig = px.box(dataframe,
    #              x="result",
    #              y="FLAG_PHONE",
    #              color="TARGET",
    #              notched=True)
    # st.plotly_chart(fig)

    # st.header("OCCUPATION_TYPE / EXT_SOURCE_3 / target")
    # fig = px.box(application, x="OCCUPATION_TYPE", y="EXT_SOURCE_3", color="TARGET", notched=True)
    # st.plotly_chart(fig)

    # st.header("OCCUPATION_TYPE / EXT_SOURCE_2 / target")
    # fig = px.box(application, x="OCCUPATION_TYPE", y="EXT_SOURCE_2", color="TARGET", notched=True)
    # st.plotly_chart(fig)

    # st.header("Répartition Homme/Femme")
    # fig = px.scatter(application, x="CODE_GENDER", y="AGE", color="TARGET")
    # st.plotly_chart(fig)

    # repart_HF = application.groupby(
    #     by=['CODE_GENDER', 'TARGET'])['SK_ID_CURR'].count()
    # st.header("Répartition Homme/Femme")
    # fig = px.bar(repart_HF, x="AGE", y="TARGET", color="CODE_GENDER")
    # st.plotly_chart(fig)

    ### graph 3 ###
    #Fonction pour les graphes
    # def count_graph(df, x, color):
    group = pd.DataFrame(
        application.groupby(['CODE_GENDER', 'TARGET'])['SK_ID_CURR'].count()).reset_index()
    group.rename(columns={'SK_ID_CURR': 'count'}, inplace=True)
    succes = group[group['TARGET'] == 0]
    succes['Pourcentage'] = succes['count'] / succes['count'].sum() * 100
    Non_succes = group[group['TARGET'] == 1]
    Non_succes['Pourcentage'] = Non_succes['count'] / Non_succes['count'].sum(
    ) * 100
    for_fig = pd.concat([succes, Non_succes])

    fig = px.bar(for_fig, x='TARGET', y='Pourcentage', color='CODE_GENDER')
    fig.update_xaxes(type='category', title_text="Succès de remboursement")

    # return for_fig, fig
    # for_fig, fig = count_graph(application, 'success', 'CODE_GENDER')
    st.write(fig)

    ### graph 2 ###
    group = pd.DataFrame(
        application.groupby(['NAME_FAMILY_STATUS',
                             'TARGET'])['SK_ID_CURR'].count()).reset_index()
    group.rename(columns={'SK_ID_CURR': 'count'}, inplace=True)
    succes = group[group['TARGET'] == 0]
    succes['Pourcentage'] = succes['count'] / succes['count'].sum() * 100
    Non_succes = group[group['TARGET'] == 1]
    Non_succes['Pourcentage'] = Non_succes['count'] / Non_succes['count'].sum(
    ) * 100
    for_fig = pd.concat([succes, Non_succes])

    fig = px.bar(for_fig,
                 x='TARGET',
                 y='Pourcentage',
                 color='NAME_FAMILY_STATUS')
    fig.update_xaxes(type='category', title_text="Succès de remboursement")
    st.write(fig)
    ### graph 3 ###
    group = pd.DataFrame(
        application.groupby(['OCCUPATION_TYPE',
                             'TARGET'])['SK_ID_CURR'].count()).reset_index()
    group.rename(columns={'SK_ID_CURR': 'count'}, inplace=True)
    succes = group[group['TARGET'] == 0]
    succes['Pourcentage'] = succes['count'] / succes['count'].sum() * 100
    Non_succes = group[group['TARGET'] == 1]
    Non_succes['Pourcentage'] = Non_succes['count'] / Non_succes['count'].sum(
    ) * 100
    for_fig = pd.concat([succes, Non_succes])

    fig = px.bar(for_fig, x='TARGET', y='Pourcentage', color='OCCUPATION_TYPE')
    fig.update_xaxes(type='category', title_text="Succès de remboursement")

    # return for_fig, fig
    # for_fig, fig = count_graph(application, 'success', 'CODE_GENDER')
    st.write(fig)


    # px.pie(dataframe, names='result', values='TARGET', title='repartition')
    # st.write(dataframe.head(5)[['SK_ID_CURR', 'TARGET', 'PREDICTED', 'result']])
    # # pie_chart = px.pie()
    # pie_chart = px.pie(data_frame=dataframe,
    #                    values=['CLASSIFICATION'],
    #                    names='CLASSIFICATION',
    #                    title='Taux de prêts accoordés et honorés')
    # X_infos_client, values="TARGET", names="Issue Reported")
    # col1.plotly_chart(pie_chart,use_container_width = True)



    # bar_chart = px.histogram(application, x="FLAG_PHONE", color="TARGET")
    # col2.plotly_chart(bar_chart,use_container_width = True)

    # st.plotly_chart(pie_chart)

    X1 = dataframe[dataframe['SK_ID_CURR'] == id_input]

    # train model with the 20 features
    # X = X1[
    #         [
    #         'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
    #         'PAYMENT_RATE', 'DAYS_REGISTRATION', 'INCOME_CREDIT_PERC',
    #         'ANNUITY_INCOME_PERC', 'AMT_ANNUITY', 'DAYS_LAST_PHONE_CHANGE',
    #         'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'INCOME_PER_PERSON',
    #         'DAYS_EMPLOYED_PERC', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE',
    #         'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
    #         'AMT_REQ_CREDIT_BUREAU_YEAR'
    #     ]]

    X = X1[[
        'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'PAYMENT_RATE',
        'FLAG_PHONE'
    ]]

    # Variables globales
    feat_imp = feature_importance(dataframe)
    impPlot(feat_imp[:20], 'RFC')

def page3():
    id_input = st.session_state.client
    st.header("Informations du client")
    st.write("Transparence des informations du client  ", id_input)

    # Pour les informations du client
    examples_file = 'new_train_cleaned_application.csv' #'application_API.csv'
    application, liste_id = chargement_data(examples_file)
    application = application[~((application['EXT_SOURCE_1'].isnull()))]
    # application.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_infos_client = application[application['SK_ID_CURR'] == id_input]
    st.write(X_infos_client)

    # réalimenter X2 avec les variables saisies
    # Saisie des informations Client dans X2 pour prédiction nouvelle

    X2 = dataframe[dataframe['SK_ID_CURR'] == id_input]

    EXT_SOURCE_1 = st.slider("EXT_SOURCE_1", 0.1, 1.0, 0.1)
    X2['EXT_SOURCE_1'] = EXT_SOURCE_1

    EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", 0.1, 1.0, 0.1)
    X2['EXT_SOURCE_2'] = EXT_SOURCE_2

    EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", 0.1, 1.0, 0.1)
    X2['EXT_SOURCE_3'] = EXT_SOURCE_3

    AGE_TRANCHE = st.selectbox("Tranche d'âge",options=['18-25', '26-35', '35-45', '45+'])
    if AGE_TRANCHE == '18-24':
        X2['DAYS_BIRTH'] = 20*-360
    if AGE_TRANCHE =='25-34':
        X2['DAYS_BIRTH'] = 30*-360
    if AGE_TRANCHE =='35-45':
        X2['DAYS_BIRTH'] = 40*-360
    if AGE_TRANCHE =='45+':
        X2['DAYS_BIRTH'] = 50*-360
    else:
        X2['DAYS_BIRTH'] = 0

    PHONE_CHANGED = st.radio("Nouveau téléphone", options=['Oui', 'Non'])
    if PHONE_CHANGED == 'Oui':
        X2['FLAG_PHONE'] = 1
    else:
        X2['FLAG_PHONE'] = 0

    # DAYS_LAST_PHONE_CHANGE = st.slider("Date dernier achat téléphone (en année)", 1, 10, 1)
    # X2['DAYS_LAST_PHONE_CHANGE'] = DAYS_LAST_PHONE_CHANGE


    # if with_API:
    #     result = prediction(X2)
    # else:
    #     result = int(json.loads(request_prediction(LRSMOTE_URI, X2).content)["prediction"])

    # X3 = X2[[
    #         'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
    #         'PAYMENT_RATE', 'DAYS_REGISTRATION', 'INCOME_CREDIT_PERC',
    #         'ANNUITY_INCOME_PERC', 'AMT_ANNUITY', 'DAYS_LAST_PHONE_CHANGE',
    #         'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'INCOME_PER_PERSON',
    #         'DAYS_EMPLOYED_PERC', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE',
    #         'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
    #         'AMT_REQ_CREDIT_BUREAU_YEAR'
    #     ]]
    X3 = X2[[
        'EXT_SOURCE_3', 'OBS_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_1', 'PAYMENT_RATE',
        'FLAG_PHONE'
    ]]

    if with_API:
        transparence = prediction(X3)
        probability = pd.DataFrame(model.predict_proba(X3))
    else:
        result = json.loads(request_prediction(LRSMOTE_URI,
                                          X3).content)
        transparence = int(result["prediction"])
        probability = list(
            eval(result['probability'].strip('[[]]').replace(' ', ',')))
        probability = pd.DataFrame(probability).T.copy()

    if transparence == 1:
        pred = 'rejeté'
    else:
        pred = 'approuvé'

    if "approuvé" in pred:
        st.success(
            '**Crédit {}** pour le **client {}** qui a une probabilité de **non remboursement de {}%**'
            .format(pred, id_input, round(probability.iloc[0][1] * 100,
                                          2)))  #, icon="✅")
    else:
        st.error(
            '**Crédit {}** pour le **client {}** qui a une probabilité de **non remboursement de {}%**'
            .format(pred, id_input, round(probability.iloc[0][1] * 100,
                                          2)))  #, icon="🚨")


    st.write('Probabilité d"appartenance aux classes : ', probability)


my_dict = {
    "Calcul du risque": main_page,
    "Interprétabilité": page2,
    "Transparence": page3,
}

keys = list(my_dict.keys())

selected_page = st.sidebar.selectbox("Sélectionner une page", keys)
my_dict[selected_page]()

# if __name__ == '__main__':
#     main_page()
