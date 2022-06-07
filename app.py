#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import base64

import plotly.express as px 
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import dash
from dash import Dash, dcc, html, Input, Output, State
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


# In[2]:


image_filename =  'VZLogo1.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

data_source = ["Car Price Dataset", "House Price Dataset", "Life Expectancy Dataset", "Fraud Dataset",
               "Fruit Dataset", "Cancer Dataset"]
fea_type = [""]
target_dict = {"Car": "price", "House": "price", "Life": "Life expectancy", 
               "Cancer": "class", "Fruit": "class", "Fraud": "Class"}
alpha_list =[0, 0.01, 0.1, 0.5, 1, 10, 100, 1000]
classname_list= {'Cancer': ['positive', 'negative'],'Fraud': ['positive', 'negative'], "Fruit": ['apple', 'mandarin', 'orange', 'lemon']} 


# In[3]:


def read_data(data):
    filename = data.lower() + '.csv'
    df = pd.read_csv(filename, index_col = 0)
    df.columns = df.columns.str.strip()
    return df


# In[4]:


def get_summary(data,activetab):    
    df = read_data(data)
    if(activetab =='classification'):
        key_name = ['Data Set Name', 'Target Attibute (Variable)', 'Number of observations', 'Number of Classes',
                          'Number of Categorical attibutes', 'Number of Numerical Attributes']
        num_class = len(df.iloc[:,-1].unique())
        num_cat = len(df.select_dtypes(include=['object']).columns)
        value_list = [[ds for ds in data_source if data in ds][0], target_dict[data], 
                      df.shape[0], num_class, num_cat, df.shape[1]- num_cat]        
    else:
        key_name = ['Data Set Name', 'Target Attibute (Variable)', 'Number of observations', 
                      'Number of Categorical attibutes', 'Number of Numerical Attributes']
        num_cat = len(df.select_dtypes(include=['object']).columns)
        value_list = [[ds for ds in data_source if data in ds][0], target_dict[data], df.shape[0], num_cat, df.shape[1]- num_cat]
    
    dfsummary = pd.DataFrame({'Properties': value_list}, index = key_name)
    dfsummary.reset_index(inplace = True)
    
    return dfsummary


# In[5]:


def feature_selection(df, target):
    ## data cleaning
    df.replace("?", np.nan, inplace = True)
    df.replace(" ", np.nan, inplace = True)
    # remove rows that are missing 'target'
    df.dropna(subset=[target], axis=0, inplace=True)
    
    # missing columns
    for col in df.columns[df.isnull().any()].tolist():
        if(df[col].dtype=='object'):
            freq_ = df[col].value_counts().idxmax()
            df[col].replace(np.nan, freq_, inplace=True)
        else:
            avg_ = df[col].mean(axis=0)        
            df[col].replace(np.nan, avg_, inplace=True)
    # category to numerical
    category_list = df.select_dtypes(include=['object']).columns.tolist()
    df_c = df[category_list].copy()
     
    le = LabelEncoder()
    for col in category_list:
        df_c[col+'_cls'] =  le.fit_transform(df_c[col])

    df_c.drop(columns = category_list, inplace = True)
    df1 = pd.concat([df,df_c], axis = 1)
    df1.drop(columns = category_list, inplace = True)
    
    tf = abs(df1.corr().loc[target]).sort_values(ascending = False)[1:df.shape[1]//2].to_frame()       
    top_features = df1.corr().loc[target].to_frame()
    top_features = top_features.loc[list(tf.index.values)]
    
    top_features.reset_index(inplace = True)
    top_features.rename(columns = {'index' : 'feature', target: 'score' }, inplace = True)
    
    return [df1, top_features]


# In[6]:


def decision_tree(X,y, trsize):
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-trsize, random_state = 0)
    
    clf = DecisionTreeClassifier(max_depth = 5, random_state = 0)
    clf.fit(x_train, y_train)
    f_important = clf.feature_importances_
    
    df1 = pd.DataFrame({'Feature_name': x_train.columns, 'Score': f_important})
    df1.sort_values(by='Score', inplace = True, ascending = False)
    df1 = df1[df1['Score']>0]
    return df1


# In[7]:


def evaluation_score(y_test, Ypred_test, class_name):
    
    mat = confusion_matrix(y_test, Ypred_test)
    df_mat = pd.DataFrame(mat, columns = class_name, index = class_name)
    df_mat.reset_index(inplace = True)
    
    report = classification_report(y_test, Ypred_test,output_dict=True, target_names= class_name)    
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(2) 
    df_report.reset_index(inplace = True)
    df_report.rename(columns = {'index':''}, inplace = True)
    
    return df_mat, df_report


# In[8]:


def classifer_graph(X,x_train,y_train, x_test, y_test, ypred_test, data_name, class_name):
    if(len(class_name)>2): #multi class
        classname_map = {d:class_name[d-1] for d in np.arange(1,len(class_name)+1)}
        fig2 = go.Figure()
        
    else:
        precision, recall, thresholds = precision_recall_curve(y_test, ypred_test)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, ypred_test)
        if(precision[0]>0.2):
            precision = np.insert(precision, 0,0, axis = 0)
            recall = np.insert(recall, 0,1, axis = 0)
        classname_map = {1:class_name[0], 0:class_name[1]}
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x = precision, y=recall, name = 'Precision-Recall Curve'))
        fig2.add_trace(go.Scatter(x = fpr_lr, y=tpr_lr, name = 'ROC Curve'))
        fig2.update_layout(title= 'Precision-Recall and ROC curve for ' + data_name + ' Detection',font_size=16) 
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig2.update_layout(yaxis_title = 'Precision/True Positive Rate', font_size=14, yaxis_range = [0,1])
        fig2.update_layout(xaxis_title = 'Recall/False Positive Rate', xaxis_range = [0,1])
        fig2.update_layout(legend=dict(yanchor="bottom", y=0.1, xanchor="center", x=0.5))
        
        
    text_str = list(y_test.map(classname_map))[0:5]
    subset_x = pd.DataFrame({X.columns.values[0]:x_train[:,0],X.columns.values[1]:x_train[:,1], 
                             'class_name': y_train.map(classname_map)})

    fig1 = px.scatter(subset_x, x = subset_x.columns.values[0], y = subset_x.columns.values[1], 
                     color = subset_x.columns.values[2])

    fig1.add_trace(go.Scatter(x=x_test[0:5,0], y = x_test[0:5,1], text = text_str, marker_color = list(y_test[0:5]),
                             mode='markers',marker_size = 10, name = "Testing data"))

    fig1.update_layout(title= 'k-nn Classification for ' + data_name + ' using ' + str(X.shape[1]) + ' features.', font_size=16) 
    fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
    fig1.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
        
    return fig1, fig2


# In[9]:


def knn_classifier(X,y, data_name, class_name, trsize, k):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-trsize, random_state = 0)
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)

    Ypred_train = knn.predict(x_train)
    Ypred_test  = knn.predict(x_test)   
    
    df_mat, df_report = evaluation_score(y_test, Ypred_test, class_name)
    fig1, fig2 = classifer_graph(X,x_train,y_train, x_test, y_test, Ypred_test, data_name, class_name)
    
    return df_mat, df_report, fig1, fig2


# In[10]:


def draw_regression_fig(x_train, x_test, y_train, y_test, x_model, y_model, fea, target, model):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = x_train[fea], y=y_train[target], opacity=0.85, mode = 'markers', name = 'Training data'))
    fig.add_trace(go.Scatter(x = x_test[fea], y=y_test[target], opacity=0.85, mode = 'markers', name = 'Testing data'))
    fig.add_trace(go.Scatter(x = x_model, y=y_model.flatten(), name = model + ' Model'))
    fig.update_layout(title= model + ' Regression Model for ' + target + ' Prediction using ' + fea)
    fig.update_layout(xaxis_title = "Feature: " + fea, yaxis_title = target)
    
    return fig


# In[11]:


def linear_model(X,y, fea, target, trsize):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-trsize, random_state = 0)
    
    L_model = LinearRegression()
    L_model.fit(x_train, y_train)
    
    Ypred_train = L_model.predict(x_train)
    Ypred_test  = L_model.predict(x_test)

    x_model = np.linspace(X[fea].min(), X[fea].max(), 100)
    Ypred_model = L_model.predict(x_model.reshape(-1,1))
    fig = draw_regression_fig(x_train, x_test, y_train, y_test, x_model, 
                              Ypred_model, fea, target, 'Linear')
    
    rscore_tr = r2_score(y_train,Ypred_train) #true first and then predit
    rscore_ts = r2_score(y_test, Ypred_test)
    
    return [rscore_tr, rscore_ts, fig]


# In[12]:


def poly_model(X,y, fea, target, trsize, degree):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-trsize, random_state = 0)
    
    Input=[('scale',StandardScaler()),
               ('polynomial', PolynomialFeatures(degree = degree, include_bias=False)),
               ('model',LinearRegression())]

    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    pipe=Pipeline(Input)
    pipe.fit(x_train,y_train)
    Ypred_train =pipe.predict(x_train)
    Ypred_test =pipe.predict(x_test)
    
    x_model = np.linspace(X[fea].min(), X[fea].max(), 100)
    Ypred_model = pipe.predict(x_model.reshape(-1,1))
    fig = draw_regression_fig(x_train, x_test, y_train, y_test, x_model, Ypred_model, fea, target, 'Polynomial')
    
    rscore_tr = r2_score(y_train,Ypred_train) #true first and then predit
    rscore_ts = r2_score(y_test, Ypred_test)
    
    return [rscore_tr, rscore_ts, fig]


# In[13]:


def ridge_model(X,y, fea, target, trsize,degree, alpha):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-trsize, random_state = 0)
       
    pr=PolynomialFeatures(degree=degree)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    
    ridgeReg = Ridge(alpha=alpha, normalize=True)
    ridgeReg.fit(x_train_pr,y_train)
    
    Ypred_train = ridgeReg.predict(x_train_pr)
    Ypred_test  = ridgeReg.predict(x_test_pr)

    x_model = np.linspace(X[fea].min(), X[fea].max(), 100)
    x_model_pr =  pr.fit_transform(x_model.reshape(-1,1))
    Ypred_model = ridgeReg.predict(x_model_pr)
    
    fig = draw_regression_fig(x_train, x_test, y_train, y_test, x_model, Ypred_model, fea, target, 'Ridge')
    
    rscore_tr = r2_score(y_train,Ypred_train) #true first and then predit
    rscore_ts = r2_score(y_test, Ypred_test)
    
    return [rscore_tr, rscore_ts, fig]


# In[14]:


def lasso_model(X,y, fea, target, trsize,degree, alpha):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1-trsize, random_state = 0)
    
    pr=PolynomialFeatures(degree=degree)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    
    lassoReg = Lasso(alpha=alpha, max_iter=20000, normalize=True)
    lassoReg.fit(x_train_pr,y_train)        
    
    Ypred_train = lassoReg.predict(x_train_pr)
    Ypred_test  = lassoReg.predict(x_test_pr)

    x_model = np.linspace(X[fea].min(), X[fea].max(), 100)
    x_model_pr =  pr.fit_transform(x_model.reshape(-1,1))
    Ypred_model = lassoReg.predict(x_model_pr)
    
    fig = draw_regression_fig(x_train, x_test, y_train, y_test, x_model, Ypred_model, fea, target, 'Lasso')
    
    rscore_tr = r2_score(y_train,Ypred_train) #true first and then predit
    rscore_ts = r2_score(y_test, Ypred_test)
    
    return [rscore_tr, rscore_ts, fig]


# In[15]:


app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME])

app.title = "Machine Learning Models"
server = app.server


# In[16]:


app.layout = dbc.Container(
    [
        ## Header        
        html.Div([
            html.Div(
            html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()), height = 150)),
            html.Div([
                html.Br(),
                html.Br(),
                dbc.Button("Dashboard to Interpret Machine Learning Models", color="primary", size = 'lg'),],
            style={"margin-left": "55px", 'width': '60%'}),
        ],style={'display':'flex'}),        
        # Tabs 
        html.Br(),
        dbc.Tabs([
            dbc.Tab(label="Regression", tab_id="regression", active_label_style={"color": "#FB79B3"}),
            dbc.Tab(label="Classification", tab_id="classification"),
            ], id="tabs", active_tab="regression",
        ),        
        
        html.Br(),         
        
        html.Div([
        
            html.Div([
            html.Label(['Select Dataset:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(
                id = 'data-dd', options=[{'label':data,'value':data.split(' ')[0]} for data in data_source],           
                value=data_source[0].split(' ')[0],
                style={'border': '#00ff8b','color': '#5e34eb','borderStyle':'dashed','height':'50px','width': '350px'} 
            ), ]),
            
            
            html.Div([
            html.Label(['Select Feature :'], style={'font-weight': 'bold', "text-align": "center"}),   
            dcc.Dropdown(
                id = 'fea-dd', options=[{'label':fea,'value':fea} for fea in fea_type],           
                value=fea_type[0],
                style={'border': '#ff5b3d','color': '#ff34eb', 
                       'borderStyle':'dashed','height':'50px','width': '350px'} 
            ), ],  style={"margin-left": "55px"}),   
            
            html.Div([  
            html.Label(['Select Training Size:'], style={'font-weight': 'bold', "text-align": "center"}), 
            dcc.Dropdown(
                id='trsize-dd', options=[{'label': str(i)+'%', 'value': i} for i in range(0,100,5)], 
                value = 70, 
                style={'border': '#ff5b3d','color': '#ff34eb', 
                       'borderStyle':'dashed','height':'50px','width': '250px'} 
            ), ],style={"margin-left": "55px"}), 
           
        ],style = {'display': 'flex'}),
        
        html.Br(),
        html.Div([
            dbc.Button("Data Information", id = 'dataload-bt', color="info"),
            dbc.Button("Feature Selection", id = 'feaselect-bt', color="success"),
            dbc.Button("Regression Modelling", id = 'model-bt', color="warning"),
            dbc.Button("Run K-NN Classifier", id = 'knn-bt', color="primary"),],
            className="d-grid gap-3 d-md-flex justify-content-md-left", style = {'display': 'flex'}),
        
        html.Br(),
        html.Br(),
        dbc.Collapse(
            html.Div([
                html.Div([html.Label("Summary of Data-set"), html.Div(id="table1")]),
                html.Div([html.Label("Data-set Sample",style={"margin-left": "55px"}), 
                          html.Div(id="table2", style={"margin-left": "55px"}),])                
            ], style={'display':'flex'}),            
            id = 'dataload-col-id', is_open = False),
        
        dbc.Collapse(html.Div(dcc.Graph(id='fea-fig'),style={'width':'95%'}), id = 'feature-col-id', is_open = False),
               
        dbc.Collapse([
            html.Div(
                [
                    html.Div(
                        [html.Label(['Number of Neighbors:'], style={'font-weight': 'bold', "text-align": "center"}), 
                         dcc.Slider(id='k-slider', min = 1, max = 20, step = 1, value=5,
                                    marks={i: str(i) for i in range(1,21,1)}),
                         html.Div(dcc.Graph(id='knn-fig1')),
                        ],style = {'width': '70%'}),
                      
                     html.Div([
                         html.Br(), html.Br(),
                         html.Label(["Confusion Matrix (Testing Set)"], style={'font-weight': 'bold', "text-align": "center"}),                                
                         html.Div(id="con_table1")], style = {'width': '30%'}),
                    
                ],style = {'display': 'flex'}),
            
            html.Div([
                html.Div([
                    html.Label(["Evaluation Matrix (Testing Set)"], style={'font-weight': 'bold', "text-align": "center"}),                                
                    html.Div(id="report1")], style = {'width': '50%'}),                 
                html.Div(dcc.Graph(id='knn-fig2')) 
                ], style = {'display': 'flex'}),
            ],id = 'knn-col-id', is_open = False),        
      
                
        dbc.Collapse([
            html.Div([                
                html.Div([
                    html.Div([
                        html.Div([
                            dbc.Alert("R-Score (Training): ", id="tr_Rscore", color = 'success', is_open=True),
                            dbc.Alert("R-Score (Testing): ", id="ts_Rscore", 
                                      color = 'danger', is_open=True, style={"margin-left": "55px"}),                        
                        ], style = {'display':'flex'}),                   
                        dcc.Graph(id='plot1'),
                    ]),
                    html.Div([
                        html.Div([
                            dbc.Alert("R-Score (Training): ", id="tr_Rscore2", color = 'success', is_open=True),
                            dbc.Alert("R-Score (Testing): ", id="ts_Rscore2", 
                                      color = 'danger', is_open=True, style={"margin-left": "55px"}),                        
                        ], style = {'display':'flex'}),   
                        html.Label(['Polynomial Degree:'], style={'font-weight': 'bold', "text-align": "center"}), 
                        dcc.Slider(id='degree-slider', min = 1, max = 20, step = 1, value=5,
                              marks={i: str(i) for i in range(1,21,1)}),
                        dcc.Graph(id='plot2'),
                    ]),                    
                    ], style = {'display':'flex'}),
                html.Div([                      
                    dbc.Label("Alpha Value"),
                    dbc.RadioItems(id = "alpha-radio", 
                                   options=[{'label':alpha,'value':alpha} for alpha in alpha_list], 
                                   value=0, inline=True),
                ],style={'width':'50%'}), 
                html.Div([
                    html.Div([
                        html.Div([
                            dbc.Alert("R-Score (Training): ", id="tr_Rscore3", color = 'success', is_open=True),
                            dbc.Alert("R-Score (Testing): ", id="ts_Rscore3", 
                                      color = 'danger', is_open=True, style={"margin-left": "55px"}),                        
                        ], style = {'display':'flex'}),  
                        dcc.Graph(id='plot3'), 
                    ]),
                    
                    html.Div([
                        html.Div([
                            dbc.Alert("R-Score (Training): ", id="tr_Rscore4", color = 'success', is_open=True),
                            dbc.Alert("R-Score (Testing): ", id="ts_Rscore4", 
                                      color = 'danger', is_open=True, style={"margin-left": "55px"}),                        
                        ], style = {'display':'flex'}),
                        dcc.Graph(id='plot4'), 
                    ]),                    
                    ], style = {'display':'flex'}),
            ]),
        ], id="model-coll-id", is_open=False)
    ]
)


# In[17]:


def regression_model(data, fea, trsize, degree, alpha):  
    df = read_data(data)
    target = target_dict[data]
    
    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = go.Figure()
    fig4 = go.Figure()
    rtr1, rtr2, rtr3, rtr4 = [0,0,0,0]
    rts1, rts2, rts3, rts4 = [0,0,0,0]
    rscore1 = [rtr1, rtr2, rtr3, rtr4]
    rscore2 = [rts1, rts2, rts3, rts4]
    
    if target in df.columns.values:
        [df, top_features] = feature_selection(df, target) 
        if fea in top_features.feature.values:
            X = df[[fea]]
            y = df[[target]]
            trsize = trsize/100    
            [rtr1, rts1, fig1] = linear_model(X,y, fea, target, trsize)            
            [rtr2, rts2, fig2] = poly_model(X,y, fea, target, trsize, degree)
            [rtr3, rts3, fig3] = ridge_model(X,y, fea, target, trsize,degree, alpha)
            [rtr4, rts4, fig4]= lasso_model(X,y, fea, target, trsize, degree,alpha)
            rscore1 = [rtr1, rtr2, rtr3, rtr4]
            rscore2 = [rts1, rts2, rts3, rts4]    
            
    return [fig1, fig2, fig3, fig4, rscore1, rscore2]   


# In[18]:


@app.callback(    
    Output('model-coll-id', 'is_open'),
    [Input("tabs", "active_tab"), 
     Input("model-bt", "n_clicks"),],
    [State("model-coll-id", "is_open")],
)
def tog_modelresults(active_tab, m, is_open):
    if(active_tab == "classificatoin"):
        return False
    else:
        if m: 
            return not is_open
        else:            
            return is_open


# In[19]:


# display modelling results controlled by button
@app.callback( 
    [Output('plot1', 'figure'),
     Output('plot2', 'figure'),
     Output('plot3', 'figure'),
     Output('plot4', 'figure'),
     Output('tr_Rscore', 'children'),
     Output('tr_Rscore2', 'children'),
     Output('tr_Rscore3', 'children'),     
     Output('tr_Rscore4', 'children'),
     
     Output('ts_Rscore', 'children'),
     Output('ts_Rscore2', 'children'),    
     Output('ts_Rscore3', 'children'),     
     Output('ts_Rscore4', 'children')],
    [Input("tabs", "active_tab"),
     Input("data-dd", "value"),
     Input('fea-dd', 'value'),
     Input("trsize-dd", "value"),
     Input("degree-slider", "value"),
     Input("alpha-radio", "value")],
)

def model_result(active_tab, data, fea, trsize, degree, alpha):
    if not active_tab or data is None or trsize is None:
        raise PreventUpdate    
    else:
        if(active_tab == "regression"):
            [fig1, fig2, fig3, fig4, rscore1, rscore2] = regression_model(data, fea, trsize, degree, alpha)
            score_name1 = "R-Score (Training): "
            score_name2 = "R-Score (Testing): "        

        else:
            fig1 = go.Figure()
            fig2 = go.Figure()
            fig3 = go.Figure()
            fig4 = go.Figure()
            rscore1 = list(np.zeros(8))
            rscore2 = list(np.zeros(8))
            score_name1 = "R-Score (Training): "
            score_name2 = "R-Score (Testing): "
        
        rtr_str = [score_name1 + '{:,.2f}'.format(score) for score in rscore1]
        rts_str = [score_name2 + '{:,.2f}'.format(score) for score in rscore2]
        
        return fig1, fig2, fig3,fig4, rtr_str[0], rtr_str[1], rtr_str[2], rtr_str[3], rts_str[0], rts_str[1], rts_str[2], rts_str[3]


# In[20]:


# display table controled by button
@app.callback(    
    [Output('dataload-col-id', 'is_open'),
     Output("table1", "children"),
     Output("table2", "children")],
    [Input("tabs", "active_tab"),
     Input("dataload-bt", "n_clicks"),
     Input("data-dd", "value")],
    [State("dataload-col-id", "is_open")],
)

def toggle_dataframe(activetab, dataload, data, is_open):
    if data is None or activetab is None:
        raise PreventUpdate
    else:
        df = read_data(data)
        df1 = get_summary(data, activetab)
        df1 = df1.round(3)
        df2 = df.head(5).iloc[:,0:5].round(3)        
        table1 = dbc.Table.from_dataframe(df1, striped=True, bordered=True, hover=True)
        table2 = dbc.Table.from_dataframe(df2, striped=True, bordered=True, hover=True)
        if dataload:        
            return not is_open, table1, table2
        else:
            return is_open, table1, table2        


# In[21]:


@app.callback(
    [Output("model-bt", 'disabled'),
     Output("knn-bt", "disabled"),],     
    [Input("tabs", "active_tab")],  
)

def disable_modelbut(active_tab):
    if(active_tab == "classification"):
        return True, False
    elif(active_tab == "regression"):
        return False, True
    else:
        return False, False
    


# In[22]:


#to change the feature options controled by feature selection button
@app.callback(
    [Output("fea-fig", 'figure'),
     Output("fea-dd", "options"),
     Output("feature-col-id", 'is_open')],
    [Input("tabs", "active_tab"),
    Input("feaselect-bt", "n_clicks"),
    Input("data-dd", "value")],
    State("feature-col-id", 'is_open')
    
)
def update_features(active_tab, feaselect,data, is_open): 
    if not active_tab or data is None or not feaselect:
        raise PreventUpdate           
    else:        
        fig = go.Figure()            
        fea_options = [{'label':data,'value':data} for data in ["",""]]
        
        if(active_tab == "regression" and data in ['Car','House', 'Life']):
            df = read_data(data)     
            target = target_dict[data]
            [df1, top_features] = feature_selection(df, target) 
            fig = px.bar(top_features, x = 'feature', y = 'score', color = 'feature', 
                         title = "Correlation Score for Different Features")        
            fig.update_layout(yaxis_title = 'Correlation Score', yaxis_range = [-1,1], xaxis_title = "Feature")
            fea_options = [{'label':data,'value':data} for data in list(top_features.feature)]
            
        elif(active_tab =="classification" and data in ['Cancer','Fraud', 'Mushroom', 'Fruit']):
            df = read_data(data)     
            target = target_dict[data]            
            fea = list(df.columns.values[0:df.shape[1]-1])
            classname = classname_list[data]
            df1 = decision_tree(df[fea],df[[target]], 0.75) # call decision tree to select features            
            fea_options = [{'label':data,'value':data} for data in list(df1.Feature_name)]
            
            fig = px.bar(df1, x = 'Feature_name', y = 'Score', color = 'Feature_name', 
                         title = "Feature Selection using Decision Tree")        
            fig.update_layout(yaxis_title = 'Feature Important Score', yaxis_range = [0,1], xaxis_title = "Feature")
            
        if feaselect:
            return fig, fea_options, not is_open
        else:
            return fig, fea_options, is_open


# In[23]:


@app.callback(    
    Output('knn-col-id', 'is_open'),
    [Input("tabs", "active_tab"),
     Input("knn-bt", "n_clicks"),],
    [State("knn-col-id", "is_open")],
)
def tog_knnresults(active_tab, m, is_open):
    if(active_tab =="regression"):
        return False
    else:
        if m:       
            return not is_open
        else:  
            return is_open


# In[24]:


@app.callback(
    [Output('knn-fig1', 'figure'),
     Output('knn-fig2', 'figure'),
     Output('con_table1', 'children'),
     Output('report1', 'children'),],
     [Input("knn-bt", "n_clicks"),
     Input("tabs", "active_tab"),
     Input("data-dd", "value"),
     Input("trsize-dd", "value"),
     Input("k-slider", "value")],
)

def update_knnresult(knnbt, activetab, data, trsize, k):
    if not knnbt:
        raise PreventUpdate
    elif(data in ['Cancer','Fraud', 'Mushroom', 'Fruit'] and activetab =='classification'):
        df = read_data(data)
        target = target_dict[data]
        fea = list(df.columns.values[0:df.shape[1]-1])
        top_features = decision_tree(df[fea], df[[target]], 0.75)
        X = df[top_features.Feature_name]
        y = df[target]
        class_name = classname_list[data]
        df_mat, df_report, fig1, fig2 = knn_classifier(X,y, data, class_name, trsize/100, k)        
        table1 = dbc.Table.from_dataframe(df_mat, striped=True, bordered=True, hover=True)
        table2 = dbc.Table.from_dataframe(df_report, striped=True, bordered=True, hover=True)
    else:
        fig1 = go.Figure()
        fig2 = go.Figure()
        df_dummies = pd.DataFrame()
        table1 = dbc.Table.from_dataframe(df_dummies, striped=True, bordered=True, hover=True)
        table2 = dbc.Table.from_dataframe(df_dummies, striped=True, bordered=True, hover=True)
        
    return fig1, fig2, table1, table2


# In[25]:


#to change data source by tags
@app.callback(
    Output("data-dd", "options"),
    Input("tabs", "active_tab"),
    State("data-dd", "value")
)
def update_options(active_tab, value):
    if not active_tab:
        raise PreventUpdate
    elif(active_tab == "regression"):
        data_source = ["Car Price Dataset", "House Price Dataset", "Life Expectancy"]        
    elif(active_tab =="classification"):
        data_source = [ "Cancer Dataset", "Fraud Dataset", "Fruit Dataset"]
    else:
        data_source = ["Car Price Dataset", "House Price Dataset"]
    
    return [{'label':data,'value':data.split(' ')[0]} for data in data_source]


# In[ ]:


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)    
    url = "http://127.0.0.1:{0}".format(port)    
    app.run_server(use_reloader=False, debug=True, port=port)


# In[ ]:




