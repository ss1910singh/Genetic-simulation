import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Loading data for visualization
try:
    df = pd.read_csv('population_log.csv')
except Exception as e:
    print(f"Error reading CSV file: {e}")
    df = pd.DataFrame()

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Genetic Simulation Dashboard", style={'textAlign': 'center', 'color': '#4A90E2'}),

    html.Div([
        dcc.Dropdown(
            id='entity-type-dropdown',
            options=[{'label': et, 'value': et} for et in df['entity_type'].unique()],
            value=df['entity_type'].unique().tolist(),
            multi=True,
            placeholder="Select entity types",
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center', 'padding': '20px'}),

    # Section: Height Distribution
    html.Div([
        dcc.Graph(id='height-histogram'),
        html.P("Height Distribution: This histogram shows the distribution of height across selected entities. "
               "It helps in understanding the variation in size among different entities, providing insight into "
               "how certain environmental or genetic factors might influence growth.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Fitness Score Distribution
    html.Div([
        dcc.Graph(id='fitness-score-histogram'),
        html.P("Fitness Score Distribution: This histogram reveals the distribution of fitness scores, allowing us "
               "to observe the overall health and adaptability of entities over generations.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Mutations Distribution
    html.Div([
        dcc.Graph(id='mutation-histogram'),
        html.P("Mutations Distribution: This histogram captures the frequency of mutations across the entity population. "
               "It can indicate the genetic variability within the population, highlighting how often changes in genes occur.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Speed Distribution
    html.Div([
        dcc.Graph(id='speed-histogram'),
        html.P("Speed Distribution: Speed variation across entities is displayed here, giving insight into how "
               "different environmental or genetic conditions affect mobility.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Height vs Fitness Score
    html.Div([
        dcc.Graph(id='scatter-plot'),
        html.P("Height vs Fitness Score: This scatter plot helps us observe the correlation between height and fitness. "
               "This relationship might reveal how physical attributes impact fitness in a specific environment.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Correlation Heatmap
    html.Div([
        dcc.Graph(id='heatmap-correlation'),
        html.P("Attribute Correlation Heatmap: This heatmap shows correlations among numerical attributes, highlighting "
               "which traits tend to vary together. High correlations can indicate linked or interdependent traits.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: PCA 2D Projection
    html.Div([
        dcc.Graph(id='pca-2d-plot'),
        html.P("PCA 2D Projection: This plot reduces the dataset's complexity by summarizing multiple dimensions into "
               "two principal components, helping us visualize overall patterns and clustering tendencies among entities.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: K-Means Clustering
    html.Div([
        dcc.Graph(id='cluster-kmeans'),
        html.P("K-Means Clustering: This visualization groups entities into clusters, showing natural groupings based "
               "on traits. Observing cluster characteristics can reveal subpopulations with similar genetic traits.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Mutation Box Plot
    html.Div([
        dcc.Graph(id='mutation-box-plot'),
        html.P("Mutation Spread by Entity Type: This box plot displays the range and distribution of mutations across "
               "different entity types, helping to understand which types undergo more frequent genetic changes.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Fitness vs Age Violin Plot
    html.Div([
        dcc.Graph(id='violin-plot-fitness-age'),
        html.P("Fitness vs Age Violin Plot: This plot illustrates the distribution of fitness scores across different ages, "
               "allowing us to identify how fitness changes with age within the population.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    # Section: Lifespan vs Age Density Plot
    html.Div([
        dcc.Graph(id='density-plot-lifespan-age'),
        html.P("Lifespan vs Age Density Plot: The 2D histogram here represents the density of entities at different age "
               "and lifespan values, giving a view of population longevity and age distribution.",
               style={'textAlign': 'center', 'padding': '10px'})
    ]),

    html.Div(id='error-message', style={'textAlign': 'center', 'color': 'red'})
])

@app.callback(
    [
        Output('height-histogram', 'figure'),
        Output('fitness-score-histogram', 'figure'),
        Output('mutation-histogram', 'figure'),
        Output('speed-histogram', 'figure'),
        Output('scatter-plot', 'figure'),
        Output('heatmap-correlation', 'figure'),
        Output('pca-2d-plot', 'figure'),
        Output('cluster-kmeans', 'figure'),
        Output('mutation-box-plot', 'figure'),
        Output('violin-plot-fitness-age', 'figure'),
        Output('density-plot-lifespan-age', 'figure'),
        Output('error-message', 'children')
    ],
    [Input('entity-type-dropdown', 'value')]
)
def update_graphs(selected_entity_types):
    if df.empty:
        return [{} for _ in range(11)] + ["No data available to display."]

    filtered_df = df[df['entity_type'].isin(selected_entity_types)]
    
    if filtered_df.empty:
        return [{} for _ in range(11)] + ["No data available for the selected entity types."]

    try:
        # Data visualizations with descriptions as before
        height_hist = go.Histogram(x=filtered_df['height'], nbinsx=20, name='Height')
        fitness_hist = go.Histogram(x=filtered_df['fitness_score'], nbinsx=20, name='Fitness Score')
        mutation_hist = go.Histogram(x=filtered_df['mutations'], nbinsx=20, name='Mutations')
        speed_hist = go.Histogram(x=filtered_df['speed'], nbinsx=20, name='Speed')
        scatter_plot = go.Scatter(x=filtered_df['height'], y=filtered_df['fitness_score'],
                                  mode='markers', marker=dict(size=8, color=filtered_df['entity_type'].astype('category').cat.codes))

        numeric_df = filtered_df.select_dtypes(include=['number'])
        heatmap_corr = go.Heatmap(z=numeric_df.corr().values, x=numeric_df.columns, y=numeric_df.columns)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numeric_df.fillna(0))
        pca_plot = go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers', marker=dict(color=filtered_df['entity_type'].astype('category').cat.codes))

        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(numeric_df.fillna(0))
        kmeans_plot = go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], mode='markers', marker=dict(color=clusters))

        mutation_box = go.Box(x=filtered_df['entity_type'], y=filtered_df['mutations'], name='Mutations')
        fitness_age_violin = go.Violin(x=filtered_df['age'], y=filtered_df['fitness_score'])
        lifespan_age_density = go.Histogram2d(x=filtered_df['age'], y=filtered_df['lifespan'])

        return (
            {'data': [height_hist], 'layout': go.Layout(title='Height Distribution')},
            {'data': [fitness_hist], 'layout': go.Layout(title='Fitness Score Distribution')},
            {'data': [mutation_hist], 'layout': go.Layout(title='Mutations Distribution')},
            {'data': [speed_hist], 'layout': go.Layout(title='Speed Distribution')},
            {'data': [scatter_plot], 'layout': go.Layout(title='Height vs Fitness Score')},
            {'data': [heatmap_corr], 'layout': go.Layout(title='Attribute Correlation Heatmap')},
            {'data': [pca_plot], 'layout': go.Layout(title='PCA 2D Projection')},
            {'data': [kmeans_plot], 'layout': go.Layout(title='K-Means Clustering')},
            {'data': [mutation_box], 'layout': go.Layout(title='Mutation Spread by Entity Type')},
            {'data': [fitness_age_violin], 'layout': go.Layout(title='Fitness vs Age Violin Plot')},
            {'data': [lifespan_age_density], 'layout': go.Layout(title='Lifespan vs Age Density Plot')},
            ""
        )

    except Exception as e:
        return [{} for _ in range(11)] + [f"Error in updating graphs: {e}"]

if __name__ == '__main__':
    app.run_server(debug=True)
