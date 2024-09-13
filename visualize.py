import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# Loading data for visulization
try:
    df = pd.read_csv('population_log.csv')
except Exception as e:
    print(f"Error reading CSV file: {e}")
    df = pd.DataFrame() 

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Genetic Simulation Dashboard"),
    
    dcc.Dropdown(
        id='entity-type-dropdown',
        options=[{'label': et, 'value': et} for et in df['entity_type'].unique()],
        value=df['entity_type'].unique().tolist(),
        multi=True
    ),
    
    dcc.Graph(id='height-histogram'),
    dcc.Graph(id='fitness-score-histogram'),
    dcc.Graph(id='mutation-histogram'),
    dcc.Graph(id='speed-histogram'),
    dcc.Graph(id='scatter-plot'),
    
    html.Div(id='error-message')
])

@app.callback(
    [Output('height-histogram', 'figure'),
     Output('fitness-score-histogram', 'figure'),
     Output('mutation-histogram', 'figure'),
     Output('speed-histogram', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('error-message', 'children')],
    [Input('entity-type-dropdown', 'value')]
)
def update_graphs(selected_entity_types):
    if df.empty:
        return {}, {}, {}, {}, {}, "No data available to display."

    filtered_df = df[df['entity_type'].isin(selected_entity_types)]
    
    if filtered_df.empty:
        return {}, {}, {}, {}, {}, "No data available for the selected entity types."
    
    try:
        height_hist = go.Histogram(x=filtered_df['height'], nbinsx=20, name='Height')
        fitness_hist = go.Histogram(x=filtered_df['fitness_score'], nbinsx=20, name='Fitness Score')
        mutation_hist = go.Histogram(x=filtered_df['mutations'], nbinsx=20, name='Mutations')
        speed_hist = go.Histogram(x=filtered_df['speed'], nbinsx=20, name='Speed')
        scatter_plot = go.Scatter(
            x=filtered_df['height'], 
            y=filtered_df['fitness_score'],
            mode='markers',
            marker=dict(size=8, color=filtered_df['entity_type'].astype('category').cat.codes),
            text=filtered_df['entity_type']
        )
        
        return (
            {'data': [height_hist], 'layout': go.Layout(title='Height Distribution')},
            {'data': [fitness_hist], 'layout': go.Layout(title='Fitness Score Distribution')},
            {'data': [mutation_hist], 'layout': go.Layout(title='Mutations Distribution')},
            {'data': [speed_hist], 'layout': go.Layout(title='Speed Distribution')},
            {'data': [scatter_plot], 'layout': go.Layout(title='Height vs Fitness Score')},
            ""
        )
    except KeyError as e:
        return {}, {}, {}, {}, {}, f"Error processing data: {e}"

if __name__ == '__main__':
    app.run_server(debug=True)
