import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# -----------------------------
# App Initialization
# -----------------------------
app = dash.Dash(__name__)
server = app.server  # For deployment

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("/content/drive/MyDrive/A2M_ABCB1_extracted.csv")
df.columns = df.columns.str.strip()  # Remove extra spaces

A2M_values = df["A2M (2)"].dropna().values
ABCB1_values = df["ABCB1 (5243)"].dropna().values

global_max = max(np.max(np.abs(A2M_values)), np.max(np.abs(ABCB1_values)))

# -----------------------------
# Gene Categories
# -----------------------------
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

# -----------------------------
# Utility Functions
# -----------------------------
def compute_probabilities(values):
    """
    Computes off-target probabilities and safety percentage
    """
    D = np.mean(np.abs(values))
    D_norm = D / global_max
    risk = D_norm
    neutral = 1 - D_norm
    each = risk / 4
    probs = [each]*4 + [neutral]
    return probs, neutral * 100

# -----------------------------
# Landing Page
# -----------------------------
def landing_page():
    return html.Div(style={'fontFamily':'Arial','padding':'20px'}, children=[
        html.Header([
            html.Div("☰", style={'float':'left', 'fontSize':'24px', 'cursor':'pointer'}),  # Hamburger placeholder
            html.Div("CRISPR-GPT", style={'textAlign':'center', 'fontSize':'24px'}),
            html.Div("🔍", style={'float':'right', 'fontSize':'24px', 'cursor':'pointer'})  # Search placeholder
        ], style={'overflow':'auto', 'padding':'10px', 'borderBottom':'1px solid #ccc'}),
        
        html.Br(),

        # Introduction paragraph
        html.P(
            "CRISPR-GPT is a prototype platform demonstrating the simulation of "
            "off-target effects in gene transgenesis. This low-scale simulation "
            "does not capture real biological complexity but illustrates the working "
            "of a predictive interface."
        ),

        # Subheading
        html.H2("Off-Target Effects Probabilities", style={'marginTop':'30px'}),

        # Gene links
        html.Ul([
            html.Li(html.A("A2M", href="/a2m", style={'textDecoration':'none', 'fontSize':'18px'})),
            html.Li(html.A("ABCB1", href="/abcb1", style={'textDecoration':'none', 'fontSize':'18px'}))
        ]),

        html.Footer("Developed by Kingsuk Singha", style={'marginTop':'50px', 'textAlign':'center'})
    ])

# -----------------------------
# Gene Page
# -----------------------------
def gene_page(gene):
    values = A2M_values if gene == "A2M" else ABCB1_values
    probs, safety = compute_probabilities(values)
    categories = gene_categories[gene]

    # Determine safety color and verdict
    if safety < 30:
        color = "red"
        verdict = "Unsafe"
    elif safety < 60:
        color = "gold"
        verdict = "Moderately Safe"
    else:
        color = "green"
        verdict = "Safe"

    return html.Div(style={'fontFamily':'Arial','padding':'20px'}, children=[
        # Header
        html.Header([
            html.A("🏠 Home", href="/", style={'fontSize':'20px', 'textDecoration':'none'}),
            html.Div(f"Off-Target Effect of Transgenesis: {gene}", style={'textAlign':'center', 'fontSize':'24px'})
        ], style={'overflow':'auto', 'padding':'10px', 'borderBottom':'1px solid #ccc'}),

        html.Br(),

        # Bar graph of off-target probabilities
        dcc.Graph(
            figure=go.Figure(data=[go.Bar(x=categories, y=probs)])
                    .update_layout(title=f"{gene} Off-Target Probabilities",
                                   yaxis=dict(title="Probability", range=[0,1]))
        ),

        # Safety bar
        html.Div(style={
            "width": "100%",
            "height": "30px",
            "backgroundColor": "#ddd",
            "marginTop": "20px"
        }, children=[
            html.Div(style={
                "width": f"{safety}%",
                "height": "100%",
                "backgroundColor": color
            })
        ]),

        html.H4(f"{round(safety,2)}% safe — {verdict}", style={"color": color, "marginTop":"10px"}),

        html.Footer("Developed by Kingsuk Singha", style={'marginTop':'50px', 'textAlign':'center'})
    ])

# -----------------------------
# App Layout with Routing
# -----------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# -----------------------------
# Callback for Routing
# -----------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    path = pathname.lower()
    if path == "/a2m":
        return gene_page("A2M")
    elif path == "/abcb1":
        return gene_page("ABCB1")
    else:
        return landing_page()

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)