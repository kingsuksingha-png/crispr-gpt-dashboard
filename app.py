import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)
server = app.server

df = pd.read_csv("A2M_ABCB1_extracted.csv")
df.columns = df.columns.str.strip()

A2M_values = df["A2M (2)"].dropna().values
ABCB1_values = df["ABCB1 (5243)"].dropna().values

global_max = max(
    np.max(np.abs(A2M_values)),
    np.max(np.abs(ABCB1_values))
)

gene_categories = {
    "A2M": [
        "Immune dysregulation",
        "Protease imbalance",
        "Tissue repair alteration",
        "Cytokine signaling interference",
        "Neutral"
    ],
    "ABCB1": [
        "Drug sensitivity change",
        "Detox failure",
        "Barrier disruption",
        "Energy imbalance",
        "Neutral"
    ]
}

def compute_probabilities(values):
    D = np.mean(np.abs(values))
    D_norm = D / global_max
    risk = D_norm
    neutral = 1 - D_norm
    each = risk / 4
    probs = [each]*4 + [neutral]
    return probs, neutral * 100

app.layout = html.Div(style={'fontFamily':'Arial','padding':'20px'}, children=[
    html.Header([
        html.Div("☰", style={'float':'left', 'fontSize':'24px'}),  # Hamburger
        html.Div("CRISPR-GPT", style={'textAlign':'center', 'fontSize':'24px'}),
        html.Div("🔍", style={'float':'right', 'fontSize':'24px'})  # Search icon
    ], style={'overflow':'auto', 'padding':'10px', 'borderBottom':'1px solid #ccc'}),
    
    html.Br(),
    
    # Introduction paragraph
    html.P(
        "CRISPR-GPT is a prototype platform designed to demonstrate the working "
        "of predicting off-target effects in gene transgenesis. It is a low-scale "
        "simulation and does not fully capture real biological complexity."
    ),
    
    # =======================
    # New Subheading Added
    # =======================
    html.H2("Off-Target Effects Probabilities", style={'marginTop':'30px'}),
    
    # Gene links
    html.Ul([
        html.Li(html.A("A2M", href="/a2m", style={'textDecoration':'none'})),
        html.Li(html.A("ABCB1", href="/abcb1", style={'textDecoration':'none'}))
    ]),
    
    html.Footer("Developed by Kingsuk Singha", style={'marginTop':'50px', 'textAlign':'center'})
])

if __name__ == "__main__":
    app.run(debug=True)

def gene_page(gene):
    values = A2M_values if gene == "A2M" else ABCB1_values
    probs, safety = compute_probabilities(values)
    categories = gene_categories[gene]

    fig = go.Figure(data=[go.Bar(x=categories, y=probs)])

    if safety < 30:
        color = "red"
        verdict = "Unsafe"
    elif safety < 60:
        color = "gold"
        verdict = "Moderately Safe"
    else:
        color = "green"
        verdict = "Safe"

    return html.Div([
        dcc.Link("Home", href="/"),
        dcc.Graph(figure=fig),
        html.Div(style={
            "width": "100%",
            "height": "30px",
            "backgroundColor": "#ddd"
        }, children=[
            html.Div(style={
                "width": f"{safety}%",
                "height": "100%",
                "backgroundColor": color
            })
        ]),
        html.H4(f"{round(safety,2)}% safe — {verdict}",
                style={"color": color}),
        html.Div("Developed by Kingsuk Singha")
    ])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def route(path):
    if path == "/A2M":
        return gene_page("A2M")
    elif path == "/ABCB1":
        return gene_page("ABCB1")
    else:
        return landing_page()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
