"""
Created on Tue May 12 13:04:42 2026
Interactive Dashboard for the NBA Draft Model results
Note - open http://127.0.0.1:8050 in your browser to view
@author: Subramanya.Ganti
"""
import numpy as np
import pandas as pd

import os
import sys
import dash
from dash import dcc, html, dash_table, Input, Output #callback_context
#import plotly.express as px
import plotly.graph_objects as go

try:
    # Works in standard .py scripts
    current_file = __file__
except NameError:
    # Fallback for Jupyter/IPython
    current_file = os.path.abspath('dashboard.py')

path = os.path.dirname(current_file)
#path = "C:/Users/uttam/Desktop/Sports/basketball/bart"

YEARS = list(range(2014, 2026))

#%% Data
df = []
for y in YEARS + [max(YEARS)+1]:
    dfn = pd.read_excel(f'{path}/results.xlsx',f'{y}')
    df.append(dfn)
    
df = pd.concat(df)
df = df.sort_values(by='mean DPM', ascending=False)
df = df.rename(columns={'mean DPM':'meanDPM','all star':'allstar','all nba':'allnba','P5 DPM':'p5','median DPM':'median','P95 DPM':'p95'})
df = df[['player','team','age','season','meanDPM','rotation','starter','allstar','allnba','mvp','p5','median','p95']]
df['age'] = round(df['age'],2)
#df = df.head(250)
df = df.dropna()

def extract_validation_data(key,new_names):
    dfv = pd.read_excel(f'{path}/validation.xlsx',key, index_col=0)
    dfv = dfv.drop('sample size', axis=1)
    dfv.columns = new_names
    return dfv

df_obs_mod = extract_validation_data('summary', ['year', 'obs_rotation', 'obs_starter', 'obs_allstar', 'obs_allnba',
                                     'obs_mvp', 'mod_rotation', 'mod_starter', 'mod_allstar', 'mod_allnba', 'mod_mvp'])
df_corr = extract_validation_data('rmse', ['year', 'corr', 'rmse'])
df_rho = extract_validation_data('rho', ['year', 'meandpm', 'rotation', 'starter', 'allstar', 'allnba'])
df_tau = extract_validation_data('tau', ['year', 'meandpm', 'rotation', 'starter', 'allstar', 'allnba'])
df_brier = extract_validation_data('brier', ['year', 'rotation', 'starter', 'allstar', 'allnba'])
df_logloss = extract_validation_data('log_loss', ['year', 'rotation', 'starter', 'allstar', 'allnba'])
df_roc = extract_validation_data('roc', ['year', 'rotation', 'starter', 'allstar', 'allnba'])
df_prauc = extract_validation_data('pr', ['year', 'rotation', 'starter', 'allstar', 'allnba'])
df_prgain = extract_validation_data('gain', ['year', 'rotation', 'starter', 'allstar', 'allnba'])

#%% Constants

TIER_COLS   = ["rotation","starter","allstar","allnba","mvp"]
TIER_LABELS = {"rotation":"Rotation","starter":"Starter","allstar":"All-star","allnba":"All-NBA","mvp":"MVP"}
TIER_COLORS = {"rotation":"#378ADD","starter":"#1D9E75","allstar":"#D85A30","allnba":"#D4537E","mvp":"#7F77DD"}

GRID  = "#f0f0f0"
FONT  = "system-ui, -apple-system, sans-serif"

DIAG_OPTIONS = [
    {"label":"Observed vs Model",      "value":"obs_mod"},
    {"label":"Correlation & RMSE",     "value":"corr_rmse"},
    {"label":"Spearman Rho",           "value":"rho"},
    {"label":"Kendall Tau",            "value":"tau"},
    {"label":"Brier Score",            "value":"brier"},
    {"label":"Log Loss",               "value":"logloss"},
    {"label":"ROC AUC",                "value":"roc"},
    {"label":"PR AUC",                 "value":"prauc"},
    {"label":"PR Gain",                "value":"prgain"},
]

DIAG_DESCRIPTIONS = {
    "obs_mod":   "Actual player counts (bars) vs model-predicted expected counts (dashed line) by draft class year.",
    "corr_rmse": "Pearson correlation (purple, left axis) and RMSE (orange, right axis) between model probabilities and outcomes.",
    "rho":       "Spearman rank correlation (ρ) between model-predicted and observed outcomes by tier. Missing values indicate insufficient observations for the tier.",
    "tau":       "Kendall's Tau rank correlation between model-predicted and observed outcomes by tier. Missing values indicate insufficient observations for the tier.",
    "brier":     "Brier score measures probability calibration. Lower is better.",
    "logloss":   "Log loss penalises confident wrong predictions more heavily. Lower is better.",
    "roc":       "ROC AUC measures discriminative ability. Higher is better (1.0 = perfect). Missing all-NBA values in 2024–25 reflect zero observed outcomes.",
    "prauc":     "Precision-Recall AUC. More informative than ROC for imbalanced classes. Higher is better.",
    "prgain":    "PR Gain — a rescaled version of PR AUC that accounts for class imbalance. Higher is better.",
}

#%% Helpers
def fmt_pct(v): return f"{v*100:.1f}%"

def summary_card(label, value, color="#111"):
    return html.Div(
        style={"background":"#f5f5f3","borderRadius":"8px","padding":"10px 12px"},
        children=[
            html.Div(label, style={"fontSize":"11px","color":"#666","marginBottom":"2px"}),
            html.Div(value, style={"fontSize":"18px","fontWeight":"500","color":color}),
        ]
    )

def base_layout(xtitle, ytitle):
    return dict(
        margin=dict(l=54, r=24, t=28, b=44),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=xtitle, showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(title=ytitle, showgrid=True, gridcolor=GRID, zeroline=False),
    )

def line_trace(name, y, color, dash="solid"):
    return go.Scatter(x=YEARS, y=y, name=name, mode="lines+markers",
                      line=dict(color=color, width=2, dash=dash),
                      marker=dict(size=5))

#%% diagnostics graphs
def build_diag_fig(tab, obs_mod_tier="rotation"):
    fig = go.Figure()

    TIER_TRACES = [
        ("Rotation", "rotation", TIER_COLORS["rotation"], "solid"),
        ("Starter",  "starter",  TIER_COLORS["starter"],  "dash"),
        ("All-star", "allstar",  TIER_COLORS["allstar"],  "dot"),
        ("All-NBA",  "allnba",   TIER_COLORS["allnba"],   "longdash"),
    ]
    RANK_TRACES = [
        ("Mean DPM", "meandpm",  "#888780",               "longdashdot"),
        ("Rotation", "rotation", TIER_COLORS["rotation"], "solid"),
        ("Starter",  "starter",  TIER_COLORS["starter"],  "dash"),
        ("All-star", "allstar",  TIER_COLORS["allstar"],  "dot"),
        ("All-NBA",  "allnba",   TIER_COLORS["allnba"],   "longdash"),
    ]

    def safe_line(name, series, color, dash):
        """Drop None/NaN before plotting so lines don't break."""
        y_vals = series.tolist()
        x_vals = YEARS
        pairs  = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None and not (isinstance(y, float) and pd.isna(y))]
        if not pairs:
            return
        xs, ys = zip(*pairs)
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), name=name, mode="lines+markers",
                                 line=dict(color=color, width=2, dash=dash),
                                 marker=dict(size=5)))

    if tab == "obs_mod":
        t   = obs_mod_tier
        col = TIER_COLORS[t]
        fig.add_trace(go.Bar(name="Observed", x=YEARS, y=df_obs_mod[f"obs_{t}"],
                             marker_color=col, opacity=0.80))
        fig.add_trace(go.Scatter(name="Model", x=YEARS, y=df_obs_mod[f"mod_{t}"],
                                 mode="lines+markers",
                                 line=dict(color="#333", width=2, dash="dash"),
                                 marker=dict(size=5)))
        layout = base_layout("Draft year", f"{TIER_LABELS[t]} count")

    elif tab == "corr_rmse":
        fig.add_trace(go.Scatter(x=YEARS, y=df_corr["corr"], name="Pearson r",
                                 mode="lines+markers",
                                 line=dict(color="#7F77DD", width=2),
                                 marker=dict(size=5), yaxis="y"))
        fig.add_trace(go.Scatter(x=YEARS, y=df_corr["rmse"], name="RMSE",
                                 mode="lines+markers",
                                 line=dict(color="#D85A30", width=2, dash="dash"),
                                 marker=dict(size=5), yaxis="y2"))
        layout = base_layout("Draft year", "Pearson r")
        layout["yaxis"]["color"]  = "#7F77DD"
        layout["yaxis2"] = dict(title="RMSE", overlaying="y", side="right",
                                showgrid=False, color="#D85A30", zeroline=False)

    elif tab == "rho":
        for name, col, color, dash in RANK_TRACES:
            safe_line(name, df_rho[col], color, dash)
        layout = base_layout("Draft year", "Spearman ρ")

    elif tab == "tau":
        for name, col, color, dash in RANK_TRACES:
            safe_line(name, df_tau[col], color, dash)
        layout = base_layout("Draft year", "Kendall τ")

    elif tab == "brier":
        for name, col, color, dash in TIER_TRACES:
            safe_line(name, df_brier[col], color, dash)
        layout = base_layout("Draft year", "Brier score ↓ better")

    elif tab == "logloss":
        for name, col, color, dash in TIER_TRACES:
            safe_line(name, df_logloss[col], color, dash)
        layout = base_layout("Draft year", "Log loss ↓ better")

    elif tab == "roc":
        for name, col, color, dash in TIER_TRACES:
            safe_line(name, df_roc[col], color, dash)
        layout = base_layout("Draft year", "ROC AUC ↑ better")

    elif tab == "prauc":
        for name, col, color, dash in TIER_TRACES:
            safe_line(name, df_prauc[col], color, dash)
        layout = base_layout("Draft year", "PR AUC ↑ better")

    elif tab == "prgain":
        for name, col, color, dash in TIER_TRACES:
            safe_line(name, df_prgain[col], color, dash)
        layout = base_layout("Draft year", "PR Gain ↑ better")

    else:
        layout = base_layout("", "")

    fig.update_layout(**layout, height=380) #, font=dict(family=FONT)
    return fig

#%% App layout
_dd   = {"fontSize":"13px"}
_lbl  = {"fontSize":"12px","color":"#666","marginBottom":"4px","display":"block"}
_wrap = {"display":"flex","flexDirection":"column"}

app = dash.Dash(__name__, title="NBA Draft Board")
server = app.server

app.layout = html.Div(
    style={"fontFamily":FONT,"maxWidth":"1200px","margin":"0 auto","padding":"24px 16px"},
    children=[

        html.H1("NBA Draft Model by Uttam (x.com/FPL_Mou)",
                style={"fontSize":"22px","fontWeight":"500","marginBottom":"4px"}),
        html.P("Probabilistic forecast of the 5-year peak DARKO DPM for every prospect (Article - fplmou.substack.com/p/nba-draft-model)",
               style={"fontSize":"13px","color":"#888","marginBottom":"20px"}),

        dcc.Tabs(id="main-tabs", value="draft",
                 colors={"border":"#e0e0e0","primary":"#378ADD","background":"#fafafa"},
                 children=[

                    # ── Draft board ───────────────────────────────────────────
                    dcc.Tab(label="Draft Board", value="draft",
                            style={"fontSize":"13px"}, selected_style={"fontSize":"13px"},
                            children=[html.Div(style={"paddingTop":"16px"}, children=[

                        # Filters row
                        html.Div(
                            style={"display":"grid",
                                   "gridTemplateColumns":"repeat(auto-fit,minmax(200px,1fr))",
                                   "gap":"12px","marginBottom":"16px"},
                            children=[
                                html.Div([html.Label("Year", style=_lbl),
                                          dcc.Dropdown(id="filter-year",
                                              options=[{"label":"All years","value":"all"}]+
                                                      [{"label":p,"value":p} for p in sorted(df["season"].unique())],
                                              value="all", clearable=False, style=_dd)]),
                                html.Div([html.Label("Player", style=_lbl),
                                          dcc.Dropdown(id="filter-player",
                                              options=[{"label":"All players","value":"all"}]+
                                                      [{"label":p,"value":p} for p in sorted(df["player"].unique())],
                                              value="all", clearable=False, style=_dd)]),
                                html.Div([html.Label("Team / School", style=_lbl),
                                          dcc.Dropdown(id="filter-team",
                                              options=[{"label":"All teams","value":"all"}]+
                                                      [{"label":t,"value":t} for t in sorted(df["team"].unique())],
                                              value="all", clearable=False, style=_dd)]),
                            ]
                        ),

                        # Summary cards
                        html.Div(id="summary-cards",
                                 style={"display":"grid",
                                        "gridTemplateColumns":"repeat(auto-fit,minmax(110px,1fr))",
                                        "gap":"8px","marginBottom":"16px"}),

                        # Table
                        dash_table.DataTable(
                            id="draft-table",
                            columns=[
                                {"name":"#",          "id":"rank",    "type":"numeric"},
                                {"name":"Player",     "id":"player",  "type":"text"},
                                {"name":"School",     "id":"team",    "type":"text"},
                                {"name":"Age",        "id":"age",     "type":"numeric"},
                                {"name":"year",       "id":"season",  "type":"numeric"},
                                {"name":"Mean DPM",   "id":"meanDPM", "type":"numeric"},
                                {"name":"Rotation",   "id":"rotation","type":"numeric"},
                                {"name":"Starter",    "id":"starter", "type":"numeric"},
                                {"name":"All-star",   "id":"allstar", "type":"numeric"},
                                {"name":"All-NBA",    "id":"allnba",  "type":"numeric"},
                                {"name":"MVP",        "id":"mvp",     "type":"numeric"},
                                {"name":"P5 DPM",     "id":"p5",      "type":"numeric"},
                                {"name":"Median DPM", "id":"median",  "type":"numeric"},
                                {"name":"P95 DPM",    "id":"p95",     "type":"numeric"},
                            ],
                            data=[],
                            sort_action="native",
                            page_size=60,
                            style_table={"overflowX":"auto"},
                            style_header={
                                "fontSize":"11px","fontWeight":"500","color":"#666",
                                "backgroundColor":"#fff","borderBottom":"1px solid #e0e0e0",
                                "padding":"8px",
                            },
                            style_cell={
                                "fontSize":"13px","padding":"7px 8px",
                                "borderBottom":"1px solid #f0f0f0",
                                "fontFamily":FONT,
                                "whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis",
                            },
                            style_cell_conditional=[
                                {"if":{"column_id":"rank"},    "width":"36px","textAlign":"right","color":"#999"},
                                {"if":{"column_id":"player"},  "width":"180px","textAlign":"center","fontWeight":"500"},
                                {"if":{"column_id":"team"},    "width":"130px","textAlign":"center","color":"#666"},
                                {"if":{"column_id":"age"},     "width":"40px","color":"#666"},
                                {"if":{"column_id":"season"},    "width":"40px","textAlign":"center","color":"#999"},
                                {"if":{"column_id":"meanDPM"}, "width":"100px","textAlign":"center","fontWeight":"500"},
                                {"if":{"column_id":"p5"},      "width":"70px","textAlign":"right","color":"#555"},
                                {"if":{"column_id":"median"},  "width":"84px","textAlign":"right","color":"#999"},
                                {"if":{"column_id":"p95"},     "width":"70px","textAlign":"right","color":"#555"},
                                {"if":{"column_id":"rotation"},"width":"70px","textAlign":"right","color":TIER_COLORS["rotation"]},
                                {"if":{"column_id":"starter"}, "width":"64px","textAlign":"right","color":TIER_COLORS["starter"]},
                                {"if":{"column_id":"allstar"}, "width":"64px","textAlign":"right","color":TIER_COLORS["allstar"]},
                                {"if":{"column_id":"allnba"},  "width":"62px","textAlign":"right","color":TIER_COLORS["allnba"]},
                                {"if":{"column_id":"mvp"},     "width":"56px","textAlign":"right","color":TIER_COLORS["mvp"]},
                            ],
                            style_data_conditional=[
                                #{"if":{"filter_query":"{meanDPM_raw} >= 0","column_id":"meanDPM"},"color":"#1D9E75"},
                                #{"if":{"filter_query":"{meanDPM_raw} < 0", "column_id":"meanDPM"},"color":"#E24B4A"},
                                {"if": {"filter_query": "{meanDPM_raw} < -5", "column_id": "meanDPM"}, "color": "#E24B4A","fontWeight":"750"},
                                {"if": {"filter_query": "{meanDPM_raw} >= -5 && {meanDPM_raw} < -1", "column_id": "meanDPM"}, "color": "#B16F60","fontWeight":"750"},
                                {"if": {"filter_query": "{meanDPM_raw} >= -1 && {meanDPM_raw} < 0", "column_id": "meanDPM"}, "color": "#7F8E6B","fontWeight":"750"},
                                {"if": {"filter_query": "{meanDPM_raw} >= 0 && {meanDPM_raw} < 1", "column_id": "meanDPM"}, "color": "#4EA271","fontWeight":"750"},
                                {"if": {"filter_query": "{meanDPM_raw} >= 1", "column_id": "meanDPM"}, "color": "#1D9E75","fontWeight":"750"},
                                {"if":{"row_index":"odd"},"backgroundColor":"#fafafa"},
                            ],
                        ),
                    ])]),

                    # ── Model diagnostics ─────────────────────────────────────
                    dcc.Tab(label="Model Diagnostics", value="diagnostics",
                            style={"fontSize":"13px"}, selected_style={"fontSize":"13px"},
                            children=[html.Div(style={"paddingTop":"16px"}, children=[

                        html.Div(
                            style={"display":"flex","gap":"16px","flexWrap":"wrap",
                                   "alignItems":"flex-end","marginBottom":"16px"},
                            children=[
                                html.Div([html.Label("Metric", style=_lbl),
                                          dcc.Dropdown(id="diag-tab",
                                              options=DIAG_OPTIONS,
                                              value="obs_mod", clearable=False,
                                              style={**_dd,"width":"220px"})]),
                                html.Div(id="obs-mod-tier-wrapper", style=_wrap, children=[
                                    html.Label("Tier", style=_lbl),
                                    dcc.Dropdown(id="obs-mod-tier",
                                        options=[{"label":TIER_LABELS[c],"value":c}
                                                 for c in ["rotation","starter","allstar","allnba"]],
                                        value="rotation", clearable=False,
                                        style={**_dd,"width":"140px"}),
                                ]),
                            ]
                        ),

                        dcc.Graph(id="diag-chart", config={"displayModeBar":False},
                                  style={"height":"400px"}),

                        html.Div(id="diag-description",
                                 style={"fontSize":"12px","color":"#888","marginTop":"8px"}),
                    ])]),
                ]),
    ]
)

#%% Callback
@app.callback(
    Output("summary-cards", "children"),
    Output("draft-table",   "data"),
    Input("filter-year",    "value"),
    Input("filter-player",  "value"),
    Input("filter-team",    "value"),
)

def update_table(year, player, team):
    dff = df.copy()
    if player != "all":
        dff = dff[dff["player"] == player]
    if team != "all":
        dff = dff[dff["team"] == team]
    if year != "all":
        dff = dff[dff["season"] == year]

    # Summary cards
    n = len(dff)
    avg_dpm = dff["meanDPM"].mean() if n else 0
    cards = [
        summary_card("Prospects projected", str(n)),
        summary_card("Avg DPM", f"{avg_dpm:.2f}", "#1D9E75" if avg_dpm >= 0 else "#E24B4A"),
    ]
    for col in TIER_COLS:
        cards.append(summary_card(f"{TIER_LABELS[col]} caliber", f"{dff[col].sum():.1f}", TIER_COLORS[col]))

    # Table — format display values, keep raw meanDPM for conditional coloring
    dff = dff.reset_index(drop=True)
    dff.insert(0, "rank", range(1, len(dff)+1))
    dff["meanDPM_raw"] = dff["meanDPM"]          # used by style_data_conditional
    
    #for col in ["meanDPM","p5","median","p95"]:
    #    dff[col] = dff[col].apply(lambda v: f"{v:.2f}")
    for col in TIER_COLS:
        dff[col] = round(100*dff[col],2)#.apply(fmt_pct)
    
    return cards, dff.to_dict("records")

@app.callback(
    Output("diag-chart",            "figure"),
    Output("diag-description",      "children"),
    Output("obs-mod-tier-wrapper",  "style"),
    Input("diag-tab",               "value"),
    Input("obs-mod-tier",           "value"),
)
def update_diag(tab, obs_tier):
    tier_style = _wrap if tab == "obs_mod" else {"display":"none"}
    fig  = build_diag_fig(tab, obs_tier)
    desc = DIAG_DESCRIPTIONS.get(tab, "")
    return fig, desc, tier_style

#%% Run    
if __name__ == "__main__":
    app.run(debug=True)
    