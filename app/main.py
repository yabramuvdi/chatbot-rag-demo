import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash_bootstrap_components import Card, CardBody
from flask import Flask

import os
import subprocess
import pandas as pd

# Initialize the app
server = Flask(__name__)
server.config["SERVER_TIMEOUT"] = 120  # Timeout value in seconds (default is usually 60)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)

# Define the layout
app.layout = html.Div([
    dbc.Row([
       dbc.Col([
            html.H2("Parámetros", style={"color": "black"}),
            #### Botones para escoger si se genera el texto de resumeno no
            html.Div([
                html.Label("¿Generar texto de resumen?", style={"marginBottom": "10px"}), # add margin-bottom style here
                dcc.RadioItems(
                    id='gen-text-button',
                    options=[{'label': i, 'value': i} for i in ['Sí', 'No']],
                    value='No', # Default value
                    labelStyle={'display': 'inline-block', 'margin-right': '30px'}
                )
            ], style={"marginBottom": "30px"}),
            #### Dropdown para escoger el nivel de análisis
            html.Div([
                html.Label("Nivel de análisis", style={"marginBottom": "10px"}),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in ["Todo", "Entrevista", "Ciudad", "Grupo"]],
                    value="Entrevista"
                ),
            ], style={"marginBottom": "30px"}),
            #### Dropdown cuyo contenido está sujeto enteramente al primero dropdown
            html.Div([
                dcc.Dropdown(
                    id='conditional-dropdown',
                    # Don't set options here; they will be updated by the callback
                    value=None,
                    style={'fontSize': '10px'}
                ),
            ], style={"marginBottom": "30px"}),
            # K sources selection
            html.Div([
                html.Label("Número de fuentes a mostrar:", style={"marginBottom": "10px"}),
                dcc.Input(
                    id='k-sources',
                    type='number',
                    min=1,
                    max=50,
                    value=10,
                    style={"textAlign": "center"}, # center the input value
                ),
            ], style={"marginBottom": "30px"}),
        ], style={"backgroundColor": "#C83E4D", "padding": "30px", "height": "100vh"}, width=4),

        dbc.Col([
            html.H3("Buscador - Proyecto Cuarzo - Costa Rica"),
            dbc.InputGroup([
                dbc.Input(id="user-input", placeholder="Type your question...", type="text"),
                dbc.Button("Enviar", id="submit-button", color="primary", n_clicks=0),
            ], size="lg", style={"width": "100%", "padding-bottom": "20px"}),
            html.Div(id="chatbot-output"),
        ], style={"padding": "30px"}, width=8),
    ]),
])

#========
# Text generation callback
#========
@app.callback(
    Output("chatbot-output", "children"),
    Input("submit-button", "n_clicks"),
    Input("user-input", "value"),
    Input("gen-text-button", "value"),
    Input("category-dropdown", "value"),
    Input("conditional-dropdown", "value"),
    Input("k-sources", "value"),
)
def update_chatbot_output(n_clicks, user_input, gen_text, analysis_type, analysis_selected, k_sources):
    ctx = dash.callback_context

    # prevent update if no button was clicked
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'submit-button':
        raise PreventUpdate
    
    # # run script according to selected category
    # if file_category == "Informes":
    #     script_path = "./app/bot_informes.py"
    #     input_path = "nada"
    #     persist_directory = "./vectordbs/informes/"
    #     output_path = "./app/temp/"
    # elif file_category == "Transcripciones":
    #     script_path = "./app/bot_transcripciones.py"
    #     input_path = "nada"
    #     persist_directory = "./vectordbs/transcripciones/"
    #     output_path = "./app/temp/"
    # elif file_category == "Base de datos":
    #     return ["No hay chatbot para esta categoría todavía"]

    # define the parameters to run the "backend" script
    script_path = "./app/bot_transcripciones.py"
    output_path = "./app/temp/"
    persist_directory = "./vectordbs/transcripciones/"

    # run the script with the appropriate parameters
    subprocess.run(
        ["python3.10", script_path, gen_text, analysis_type, analysis_selected, persist_directory, output_path, user_input, str(k_sources)],
        check=True,
    )


    #----------------
    # Display ANSWER
    #----------------

    # Add whitespace to seprate the answer from the input
    whitespace_div = html.Div(style={"height": "20px"})

    # Create a CardBody with the chatbot's answer
    answer_card_body = CardBody([
        # heading of box, bold text
        html.H5("Respuesta", className="card-title", style={"fontWeight": "bold"}),
    ])

    # Wrap the CardBody in a Card component
    answer_card = Card(answer_card_body, style={"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.1)"})

    #----------------
    # Display SOURCES
    #----------------

    # read the sources from the csv file
    df_sources = pd.read_csv(output_path + "sources.csv")
    
    # create HTML paragraphs with the content of each source
    sources_content = [html.H5("Fuentes", className="card-title", style={"fontWeight": "bold"})]
    for i, row in df_sources.iterrows():
        sources_content.append(html.Div(style={"height": "10px"}))
        source_doc = row['source_doc'].split("/")[-1]
        sources_content.append(html.P(f"Documento: {source_doc} ---- {row['duration']}", className="card-text", style={"fontSize": "normal"}))
        sources_content.append(html.Div(style={"height": "5px"}))
        sources_content.append(html.P(f"{row['content']}", className="card-text", style={"fontSize": "normal"}))

    # consolidate sources into a card
    card_sources = CardBody(sources_content)

    # Wrap the CardBody in a Card component
    card_sources = Card(card_sources, style={"box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.1)"})

    return [whitespace_div, answer_card, whitespace_div, card_sources]

#========
# Callback function to filter the options in the second dropdown
#========
@app.callback(
    Output("conditional-dropdown", "options"),
    Output("conditional-dropdown", "value"),
    Input("category-dropdown", "value")
)
def update_file_dropdown(selected_category):
    
    # generate a list of relevant options according to the selected category
    # "Todo", "Entrevista", "Ciudad", "Tendero", "Grupo"
    if selected_category == "Entrevista":
        # generate a list of all the available files
        all_files = os.listdir("./data/transcripciones/limpias/")
        all_files = [file for file in all_files if ".csv" in file]
        options = [{'label': f"{f.replace('.csv', '')}", 'value': f} for f in all_files]
    elif selected_category == "Ciudad":
        options = [{'label': f"{f}", 'value': f} for f in ["SAN JOSE", "HEREDIA", "CARTAGO"]]
    elif selected_category == "Grupo":
        options = [{'label': f"{f}", 'value': f} for f in ["probador", "no_probador", "adoptador", "no_adoptador"]]
    # elif selected_category == "Ciudad - Grupo":
    #     options = [{'label': f"{f}", 'value': f} for f in ["bogota - Tendero", 
    #                                                        "Ciudad de Guatemala - Consumidor",
    #                                                        "Quetzaltenango - Tendero",
    #                                                        "Quetzaltenango - Consumidor"]]
    elif selected_category == "Todo":
        options = []
    
    # Set the initial value for the category-dropdown
    initial_value = "Entrevista"

    return options, initial_value

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
