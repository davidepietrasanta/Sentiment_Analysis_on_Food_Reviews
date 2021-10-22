import dash
import dash_html_components as html
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.linalg import norm
from collections import Counter





path = "E:\\Documents\\uni\\Bicocca\\Magistrale\\1 anno\\Data Analytics\\Progetto\\AAA"
df = pd.read_csv(path+"\\new_food.csv", index_col=False)

df.drop( ['Unnamed: 0'], axis=1, inplace=True )
#Gestisce la tab 2
mean_score_per_prod = df.groupby('productid').size().reset_index(name='count')
mean_score_per_prod['mean'] = np.array(df.groupby('productid')['score'].mean())

top_reviewers = {}
#Fuzioni di supporto
def my_extract(dic, x):
    #Ritorna key+value di un dizionario if value > x
    diz = {}
    if x == 0:
        for key in dic:
            diz[key] = {'n_review': dic[key]}
    else:
        for key in dic:
            if( dic[key] > x):
                diz[key] = {'n_review': dic[key]}
    return diz

def extract_score_and_productid(user_id, df):
  diz = {}
  for i in range(df.shape[0]): #Scorro le righe
    if df.at[i, 'userid'] == user_id:
      diz[df.at[i, 'productid']] = df.at[i, 'score']

  return diz

def extract_score(product_id, df):
  diz = {}
  for i in range(df.shape[0]): #Scorro le righe
    if df.at[i, 'productid'] == product_id:
      diz[df.at[i, 'productid']] = df.at[i, 'mean']

  return diz

def extract_item_from_dict(dict):
  arr = []
  for key in dict:
    for item in dict[key]['items_rate']:
      arr.append(str(item))

  return arr

def top_rev(N):
    freq_userid = pd.DataFrame( df['userid'].value_counts() )
    freq_userid = freq_userid.to_dict() #converts to dictionary
    freq_userid = freq_userid['userid']
    #Now we have a dict with {'user_id':n_reviews}
    
    top_reviewers = my_extract(freq_userid, N)

    for key in top_reviewers:
        top_reviewers[key]['items_rate'] = extract_score_and_productid(key, df)

    for key in top_reviewers:
        top_reviewers[key]['items_real_rate'] = {}
        for item in top_reviewers[key]['items_rate']:
            top_reviewers[key]['items_real_rate'][item] = extract_score(item, mean_score_per_prod)[item]

    dict_most_voted = dict(Counter( extract_item_from_dict(top_reviewers) ))
    
    max_dict = 0
    key_max = []
    for key in dict_most_voted:
        if dict_most_voted[key] > max_dict:
            max_dict = dict_most_voted[key]
            key_max = [key]
        elif dict_most_voted[key] == max_dict:
            key_max.append(key)

    stringa_top_review = "The products most reviewed by Top Reviewers have "
    stringa_top_review = stringa_top_review + str(max_dict)
    stringa_top_review = stringa_top_review + " reviews and are: " + str(key_max)

    for key in top_reviewers:
        rate_diff = []
        real_rate = []
        for item in top_reviewers[key]['items_rate']:
            rate_diff.append( top_reviewers[key]['items_real_rate'][item] - top_reviewers[key]['items_rate'][item] ) # x - x_approx
            real_rate.append( top_reviewers[key]['items_real_rate'][item] ) # x

        err = norm(rate_diff)/norm(real_rate) # norm(x - x_approx)/norm(x)
        top_reviewers[key]['err'] = err

    arr_item = []
    arr_err = []
    arr_n_review = []
    for item in top_reviewers:
        arr_item.append(item)
        arr_err.append(top_reviewers[item]['err'])
        arr_n_review.append(top_reviewers[item]['n_review'])  


    df_top_reviewers = pd.DataFrame({'User ID':arr_item, 'Percentage Error': np.array(arr_err) * 100, 'Number of Reviews': arr_n_review})

    return df_top_reviewers,  stringa_top_review



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

layout_tab1 = html.Div([
    html.H3('Search by word:'),
    # debounce fa in modo che si cerchi solo dopo che l'utente ha premuto "ENTER"
    dcc.Input(id='my-input', value='', type='text', debounce=True),
    html.Div(id='my-div'),
    dcc.Graph(id="box-plot"),
    dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            style_cell={'textAlign': 'left'},
            page_size=10,
            data=df.to_dict('records')

    )]) 


min_slider = 5
max_slider = 25
starting_value_slider = 10

df_tab2, stringa_top_review = top_rev(starting_value_slider)

layout_tab2 = app.layout = html.Div([
    html.H3('Select the minimum number of reviews to be a \"TOP REVIEWERS\"'),
    dcc.Slider(
            id="slider_tab2",
            min=min_slider,
            max=max_slider,
            marks={i: str(i) for i in range(min_slider, max_slider+1)},
            value=starting_value_slider),
    html.Div(id='my-div_2'),
    html.Div(id='my-div_2_perc'),
    html.Div(id='my-div_2_product'),
    dash_table.DataTable(
            id='table_tab2',
            columns=[{"name": i, "id": i} for i in df_tab2.columns],
            style_cell={'textAlign': 'left'},
            page_size=10,
            data=df_tab2.to_dict('records')

    )
])




app.layout = html.Div([
    html.H1('Food Reviews'),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Search Tab', value='tab-1-example'),
        dcc.Tab(label='Top Reviewers Tab', value='tab-2-example'),
    ]),
    html.Div(id='tabs-content-example')
])


#Gestisce le tab e il cambio di tag
@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return layout_tab1
    elif tab == 'tab-2-example':
        return layout_tab2 

#Gestisce la tab 1
@app.callback(
        Output(component_id='table', component_property='data'),
        Output(component_id='my-div', component_property='children'),
        Output(component_id="box-plot", component_property="figure"), 
        [Input(component_id='my-input', component_property='value')]
)
def update_output_div(input_value):

    print( input_value )

    #DataFrame
    input_value = str(input_value)
    temp_df = pd.DataFrame({})
    for i in range(len(df)):
        if( type(df.iloc[i].at['text']) == type('string')):
            if input_value in df.iloc[i].at['text']:
                temp_df=temp_df.append(df.iloc[i])

    len_df = len(temp_df)
    print(len_df)

    #Stringa output
    stringa = "The mean setiment for \'"+input_value+"\' is "
    if( len_df > 0 ): 
        afinn_score_medio = temp_df['afinn_norm'].mean()
        str_score = str(afinn_score_medio)[:3]
        
        if afinn_score_medio > 0:
            stringa = stringa + "positive ("+str_score+")"
        elif afinn_score_medio < 0:
            stringa = stringa + "negative ("+str_score+")"
        else:
            stringa = stringa + "neutral ("+str_score+")"
    else:
        stringa = "Nothig was found"

    print(stringa)

    #Boxplot
    if( len_df > 0 ): 
        fig = px.box(temp_df['afinn_norm']) 
    
    print("plot")

    return temp_df.to_dict('records'), stringa, fig






#Callback
@app.callback(
        Output(component_id='my-div_2', component_property='children'),
        [Input(component_id='slider_tab2', component_property='value')]
)
def update_output_div_tab2(input_value):
    input_value = str(input_value)
    return f'You\'ve entered "{input_value}"'


@app.callback(
        Output(component_id='table_tab2', component_property='data'),
        Output(component_id='my-div_2_perc', component_property='children'),
        Output(component_id='my-div_2_product', component_property='children'),
        [Input(component_id='slider_tab2', component_property='value')]
)
def update_output_div_tab2(input_value):
    data, stringa_top_review = top_rev(input_value)
    n_data = len(data)
    n_tot = len(df)
    perc = n_data / n_tot * 100
    perc = '%.3f' % perc
    string = "The percentage of Top Reviewers is "+str(perc)
    print(string)
    
    return data.to_dict('records'), string, stringa_top_review




if __name__ == '__main__':
    app.run_server(debug=True)

