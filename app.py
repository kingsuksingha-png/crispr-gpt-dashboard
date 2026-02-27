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

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content")
])

def landing_page():
    return html.Div([
        html.H2("CRISPR-GPT"),
        dcc.Link("A2M", href="/A2M", style={"display": "block"}),
        dcc.Link("ABCB1", href="/ABCB1", style={"display": "block"}),
        html.Div("Developed by Kingsuk Singha")
    ])

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
