import os
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Input, Output, State, dash_table, dcc, html

from layout_tools import *
from navbar import *
from src.utils.metrics import metricor
from src.utils.slidingWindows import find_length

external_stylesheets = [
	{
		'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
		'rel': 'stylesheet',
		'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
		'crossorigin': 'anonymous'
	},
	{
		'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
		'rel': 'stylesheet',
	},
	dbc.themes.BOOTSTRAP,
]

background_callback_manager = None
try:
	import diskcache
	from dash import DiskcacheManager

	cache = diskcache.Cache("./cache")
	background_callback_manager = DiskcacheManager(cache)
except ImportError:
	pass

dash_kwargs = {
	"meta_tags": [{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
	"external_stylesheets": external_stylesheets,
	"suppress_callback_exceptions": True,
	"title": "TheseusPlus",
}
if background_callback_manager is not None:
	dash_kwargs["background_callback_manager"] = background_callback_manager

app = dash.Dash(__name__, **dash_kwargs)


@lru_cache(maxsize=None)
def _load_merged_table(measure):
	return pd.read_csv(f"data/mergedTable_{measure}.csv")


def _filter_benchmark_df(df_in, dataset, anoma_type, ts_type):
	df_out = df_in
	if dataset not in (None, "ALL"):
		df_out = df_out.loc[df_out["dataset"] == dataset]
	if anoma_type not in (None, "ALL"):
		df_out = df_out.loc[df_out["type_an"] == anoma_type]
	if ts_type == "single":
		df_out = df_out.loc[df_out["nb_anomaly"] == 1.0]
	elif ts_type == "multiple":
		df_out = df_out.loc[df_out["nb_anomaly"] > 1.0]
	return df_out


def _resolve_paths(dataset_folder, filename):
	folder = dataset_folder
	if "NASA_" in folder:
		folder = folder.replace("NASA_", "NASA-")
		ts_name = filename.replace("SMAP", "").replace("_data.out", ".test.out")
	else:
		ts_name = filename.replace(".txt", ".out")

	ts_path = os.path.join(path_top_dataseries, folder, ts_name)
	scores_dir = os.path.join(path_top_anoamly_score, folder)
	return folder, ts_name, ts_path, scores_dir


def _normalize_scores(scores):
	scores = np.asarray(scores, dtype=float)
	denom = scores.max() - scores.min()
	if denom == 0:
		return scores
	return (scores - scores.min()) / denom


NECPD_METHODS = {
	"NECPD70": 70,
	"NECPD100": 100,
}
ARIMA_METHOD = "ARIMA"
ARIMA_ORDER = (1, 1, 1)
NECPD_DATA_ROOT = os.path.join("data", "new_algs")
_NECPD_MEASURES = (
	"AUC_ROC",
	"AUC_PR",
	"Precision",
	"Recall",
	"F",
	"Precision@k",
	"Rprecision",
	"Rrecall",
	"RF",
)
_NECPD_MEASURE_IDX = {name: idx for idx, name in enumerate(_NECPD_MEASURES)}


def _necpd_score_zip_path(method_name, folder, ts_name):
	return os.path.join(NECPD_DATA_ROOT, method_name, "scores", folder, "score", f"{ts_name}.zip")


@lru_cache(maxsize=None)
def _load_necpd_metrics_map(method_name):
	metrics_path = os.path.join(NECPD_DATA_ROOT, method_name, "metrics.csv")
	if not os.path.isfile(metrics_path):
		return {}
	try:
		df_ne = pd.read_csv(metrics_path)
	except Exception:
		return {}
	if "dataset" not in df_ne.columns or "filename" not in df_ne.columns:
		return {}

	available_measures = [m for m in _NECPD_MEASURES if m in df_ne.columns]
	df_ne = df_ne[["dataset", "filename"] + available_measures]

	out = {}
	for row in df_ne.to_dict("records"):
		dataset = str(row.get("dataset"))
		filename = str(row.get("filename"))
		out[(dataset, filename)] = tuple(row.get(m, np.nan) for m in _NECPD_MEASURES)
	return out


@lru_cache(maxsize=4096)
def _necpd_metrics(ts_zip_path, window):
	if not os.path.isfile(ts_zip_path):
		return (np.nan,) * len(_NECPD_MEASURES)

	try:
		ts = pd.read_csv(ts_zip_path, compression="zip", header=None).to_numpy()
	except Exception:
		return (np.nan,) * len(_NECPD_MEASURES)

	if ts.size == 0:
		return (np.nan,) * len(_NECPD_MEASURES)

	label = np.asarray(ts[:, 1], dtype=int)
	data = ts[:, 0].astype(float)
	window = max(2, min(int(window), len(data)))

	try:
		from src.models.normalized_entropy_cpd import NormalizedEntropyCPD

		clf = NormalizedEntropyCPD(window=window, bins="ln", score="delta")
		clf.fit(data)
		score = _normalize_scores(clf.decision_scores_)
	except Exception:
		return (np.nan,) * len(_NECPD_MEASURES)

	if np.sum(label) in (0, len(label)):
		return (np.nan,) * len(_NECPD_MEASURES)

	metrics = [np.nan] * len(_NECPD_MEASURES)
	grader = metricor()

	try:
		_, _, ap = grader.metric_PR(label, score)
		metrics[_NECPD_MEASURE_IDX["AUC_PR"]] = float(ap)
	except Exception:
		pass

	try:
		L = grader.metric_new(label, score)
		if L is not None:
			metrics[_NECPD_MEASURE_IDX["AUC_ROC"]] = float(L[0])
			metrics[_NECPD_MEASURE_IDX["Precision"]] = float(L[1])
			metrics[_NECPD_MEASURE_IDX["Recall"]] = float(L[2])
			metrics[_NECPD_MEASURE_IDX["F"]] = float(L[3])
			metrics[_NECPD_MEASURE_IDX["Rrecall"]] = float(L[4])
			metrics[_NECPD_MEASURE_IDX["Rprecision"]] = float(L[7])
			metrics[_NECPD_MEASURE_IDX["RF"]] = float(L[8])
			metrics[_NECPD_MEASURE_IDX["Precision@k"]] = float(L[9])
	except Exception:
		pass

	return tuple(metrics)


base_layout = html.Div([
	dcc.Location(id="url"), 
	sidebar, 
	content,
	footer,
	],
)
app.layout = base_layout



@app.callback(
	[Output(f"page-{i}-link", "active") for i in range(1, 7)],
	[Input("url", "pathname")],
)
def toggle_active_links(pathname):
	if pathname == "/":
		# Treat page 1 as the homepage / index
		return True, False, False, False, False, False
	return [pathname == f"/page-{i}" for i in range(1, 7)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
	if pathname in ["/", "/page-1"]:
		return generate_page_1()
	elif pathname == "/page-2":
		title, fig, table_children = update_page1_overview_table("ALL", "AUC_PR", "ALL", "ALL")
		return generate_page_2(
			df,
			measures="AUC_PR",
			dataset="ALL",
			title_children=title,
			boxplot_figure=fig,
			table_children=table_children,
		)
	elif pathname == "/page-3":
		return generate_page_3(df)
	elif pathname == "/page-4":
		return generate_page_4(df)
	elif pathname == "/page-5":
		return generate_page_5()
	elif pathname == "/page-6":
		return generate_page_6()
		
	# If the user tries to reach a different page, return a 404 message
	return dbc.Container(
		[
			html.H1("404: Not found", className="text-danger"),
			html.Hr(),
			html.P(f"The pathname {pathname} was not recognised..."),
		],
		className="p-5",
	)



def toggle_modal(n1,n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.callback(
    Output("modal_page_1", "is_open"),
    [Input("open_page_1", "n_clicks"), Input("close_1", "n_clicks")],
    State("modal_page_1", "is_open"),
)(toggle_modal)

app.callback(
    Output("modal_page_2", "is_open"),
    [Input("open_page_2", "n_clicks"), Input("close_2", "n_clicks")],
    State("modal_page_2", "is_open"),
)(toggle_modal)

app.callback(
    Output("modal_page_3", "is_open"),
    [Input("open_page_3", "n_clicks"), Input("close_3", "n_clicks")],
    State("modal_page_3", "is_open"),
)(toggle_modal)


############################## page 1 ################################

def add_rect(label,data):
	anom_plt = [None]*len(data)
	len_ts = len(data)
	for i,lab in enumerate(label):
		if lab == 1:
			anom_plt[i] = data[i]
			anom_plt[min(len_ts-1,i+1)] = data[min(len_ts-1,i+1)]
	return anom_plt


@app.callback(
	[Output('stat_ts_place', 'children'),
	Output('ts_place', 'children')], 
	[Input('accuracy_tbl', 'active_cell'),Input('accuracy_tbl', 'data')])
def update_page1_timeseries_graphs(active_cell,data_cell):
	if active_cell:
		row = data_cell[active_cell['row']]
		filename = row["filename"]
		dataset_folder = df.loc[df["filename"] == filename]["dataset"].values[0]
		folder, ts_name, ts_path, scores_dir = _resolve_paths(dataset_folder, filename)

		ts = pd.read_csv(ts_path + ".zip", compression="zip", header=None).to_numpy()

		label = ts[:,1]
		data = ts[:,0].astype(float)
		orig_len = len(data)
		data_min = float(np.min(data)) if orig_len else 0.0
		data_max = float(np.max(data)) if orig_len else 0.0

		plot_idx = None
		if orig_len > LIMIT_POINT_TS:
			base_idx = np.linspace(0, orig_len - 1, num=LIMIT_POINT_TS, dtype=int)
			anom_idx = np.where(label == 1)[0]
			extras = np.concatenate([anom_idx, anom_idx - 1, anom_idx + 1]) if len(anom_idx) else np.array([], dtype=int)
			plot_idx = np.unique(np.concatenate([base_idx, extras]))
			plot_idx = plot_idx[(plot_idx >= 0) & (plot_idx < orig_len)]
			label = label[plot_idx]
			data = data[plot_idx]
			x = plot_idx.tolist()
		else:
			x = list(range(orig_len))

		scores = {}
		for method_name in methods_key:
			score_zip = os.path.join(scores_dir, method_name, "score", ts_name) + ".zip"
			if os.path.isfile(score_zip):
				scores_tmp = pd.read_csv(score_zip, compression="zip", header=None).to_numpy()
				score_vec = scores_tmp[:,0].astype(float)
				if plot_idx is not None and len(score_vec) == orig_len:
					score_vec = score_vec[plot_idx]
				scores[method_name] = score_vec

		for necpd_method, necpd_window in NECPD_METHODS.items():
			necpd_zip = _necpd_score_zip_path(necpd_method, folder, ts_name)
			if os.path.isfile(necpd_zip):
				scores_tmp = pd.read_csv(necpd_zip, compression="zip", header=None).to_numpy()
				score_vec = scores_tmp[:, 0].astype(float)
				if plot_idx is not None and len(score_vec) == orig_len:
					score_vec = score_vec[plot_idx]
				scores[necpd_method] = score_vec
				continue

			try:
				from src.models.normalized_entropy_cpd import NormalizedEntropyCPD

				window = max(2, min(int(necpd_window), len(data)))
				clf = NormalizedEntropyCPD(window=window, bins="ln", score="delta")
				clf.fit(data)
				score_vec = _normalize_scores(clf.decision_scores_)
				if plot_idx is not None and len(score_vec) == orig_len:
					score_vec = score_vec[plot_idx]
				scores[necpd_method] = score_vec
			except Exception:
				pass

		arima_zip = _necpd_score_zip_path(ARIMA_METHOD, folder, ts_name)
		if os.path.isfile(arima_zip):
			scores_tmp = pd.read_csv(arima_zip, compression="zip", header=None).to_numpy()
			score_vec = scores_tmp[:, 0].astype(float)
			if plot_idx is not None and len(score_vec) == orig_len:
				score_vec = score_vec[plot_idx]
			scores[ARIMA_METHOD] = score_vec
		elif len(data) <= 5000:
			try:
				from src.models.arima_cpd import ARIMACPD

				clf = ARIMACPD(order=ARIMA_ORDER)
				clf.fit(data)
				scores[ARIMA_METHOD] = _normalize_scores(clf.decision_scores_)
			except Exception:
				pass
		
		

		anom = add_rect(label,data)
		trace_scores = []
		trace_scores.append(go.Scattergl(
			x=x,
			y=data,
			xaxis='x',
			yaxis='y2',
			name = "Time series",
			mode = 'lines',
			line = dict(color = 'blue',width=3),
			opacity = 1
		))
		trace_scores.append(go.Scattergl(
			x=x,
			y=anom,
			xaxis='x',
			yaxis='y2',
			name = "Anomalies",
			mode = 'lines',
			line = dict(color = 'red',width=3),
			opacity = 1
		))

		for method_name in scores.keys():
			if len(scores[method_name]) != len(x):
				continue
			trace_scores.append(go.Scattergl(
				x=x,
				y=scores[method_name],
				name = "{} score".format(method_name),
				opacity = 1,
				mode = 'lines',
				fill="tozeroy",
			))



		layout = go.Layout(
			yaxis=dict(
				domain=[0, 0.4],
				range=[0,1]
			),
			yaxis2=dict(
				domain=[0.45, 1],
				range=[data_min, data_max]
			),
			#showlegend=False,
			title="{} time series snippet ({}k points maximum)".format(filename.split(".")[0], int(LIMIT_POINT_TS/1000)),
			template="simple_white",
			margin=dict(l=8, r=4, t=50, b=10),
			height=375,
			hovermode="x unified",
			xaxis=dict(
				range=[0,orig_len]
			)
		)

		fig = dict(data=trace_scores, layout=layout)

		methods_bar = methods_key + list(NECPD_METHODS.keys()) + [ARIMA_METHOD]
		to_plot = pd.DataFrame({"method": methods_bar, "value": [row.get(m) for m in methods_bar]})
		to_plot["value"] = pd.to_numeric(to_plot["value"], errors="coerce")

		fig_bar = px.bar(to_plot,x="method", y="value", category_orders={"method": methods_bar}, labels={
					 "value": "{}".format('Accuracy'),
					 "method": "{}".format('AD methods'),
				 },title="{} on {} time series".format('Accuracy',filename.split(".")[0]))
		fig_bar.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)


		return [dcc.Graph(figure=fig_bar,id='stat_ts_place_1',style={'width': '100%'})],[dcc.Graph(figure=fig,id='ts_place_1',style={'width': '100%'})]
	return None,None

@app.callback(
	[Output('title_table','children'),
	Output('boxplot_page_1','figure'),
	Output('div_table_page_1', 'children')], 
	[Input('dataset_select_page_1', 'value'),
	Input('measure_select_page_1', 'value'),
	Input('type_anom_select_page_1', 'value'),
	Input('type_ts_select_page_1', 'value'),],
	prevent_initial_call=True,
)
def update_page1_overview_table(dataset,measure,anoma_type,ts_type):
	measure_name = measure if measure is not None else "AUC_PR"
	df_new = _load_merged_table(measure_name) if measure is not None else df
	df_new = _filter_benchmark_df(df_new, dataset, anoma_type, ts_type)

	df_table = df_new[["filename"] + methods_key].copy()
	necpd_idx = _NECPD_MEASURE_IDX.get(measure_name, 0)
	extra_methods = list(NECPD_METHODS.keys()) + [ARIMA_METHOD]
	precomputed_necpd = {m: _load_necpd_metrics_map(m) for m in extra_methods}
	necpd_values = {m: [] for m in extra_methods}
	for filename, dataset_folder in zip(df_new["filename"], df_new["dataset"]):
		dataset_key = str(dataset_folder)
		filename_key = str(filename)
		for method_name in extra_methods:
			pre = precomputed_necpd[method_name].get((dataset_key, filename_key))
			necpd_values[method_name].append(pre[necpd_idx] if pre is not None else np.nan)

	for method_name, values in necpd_values.items():
		df_table[method_name] = values

	methods_view = methods_key + extra_methods
	df_table = df_table[["filename"] + methods_view].round(3)

	if dataset is None: dataset = 'ALL'
	if measure is None: measure = 'AUC_PR'
	if anoma_type is None: anoma_type = 'ALL'
	to_plot = df_table[methods_view]
	fig = px.box(to_plot[to_plot.median().sort_values(ascending=True).index],labels={
					 "value": "{}".format(measure_name),
					 "variable": "{}".format('AD methods'),
				 },title="Average {} on {} time series ({})".format(measure,dataset,anoma_type))
	fig.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)
	

	return html.H5('{} for {} time series'.format(measure_name,len(df_table))),fig,[
		dash_table.DataTable(
			df_table.to_dict("records"),
			[{"name": i, "id": i} for i in df_table.columns],
			id="accuracy_tbl",
			virtualization=True,
			fixed_rows={"headers": True},
			fill_width=False,
			style_table={
				"height": table_height,
				"overflowY": "auto",
				"overflowX": "auto",
				"width": "100%",
			},
			style_cell={
				"minWidth": "7rem",
				"width": "7rem",
				"maxWidth": "7rem",
				"whiteSpace": "nowrap",
			},
			style_cell_conditional=[
				{
					"if": {"column_id": "filename"},
					"minWidth": "16rem",
					"width": "16rem",
					"maxWidth": "16rem",
				}
			],
		)
	]


############################## page 2 ################################

@app.callback(
	[Output('stat_ts_place_comp_all','children'),
	Output('comp_place', 'children')], 
	[Input('methodX_select_page_2', 'value'),
	Input('methodY_select_page_2', 'value'),
	Input('dataset_select_page_2', 'value'),
	Input('measure_select_page_2', 'value'),
	Input('type_anom_select_page_2', 'value'),
	Input('type_ts_select_page_2', 'value'),])
def update_page2_comparison(methodX,methodY,dataset,measure,anoma_type,ts_type):
	if methodX in (None, 'ALL') or methodY in (None, 'ALL'):
		return None,None

	measure_name = measure if measure is not None else "AUC_PR"

	df_new = _load_merged_table(measure_name) if measure is not None else df
	df_new = _filter_benchmark_df(df_new, dataset, anoma_type, ts_type)
	df_new = df_new[['filename','dataset']+methods_key]

	if dataset is None: dataset = 'ALL'
	if measure is None: measure = 'AUC_PR'
	if anoma_type is None: anoma_type = 'ALL'

	necpd_needed = [m for m in (methodX, methodY) if m in NECPD_METHODS or m == ARIMA_METHOD]
	if necpd_needed:
		necpd_idx = _NECPD_MEASURE_IDX.get(measure_name, 0)
		df_new = df_new.copy()
		for method_name in dict.fromkeys(necpd_needed):
			values = []
			precomputed_necpd = _load_necpd_metrics_map(method_name)
			for filename, dataset_folder in zip(df_new["filename"], df_new["dataset"]):
				pre = precomputed_necpd.get((str(dataset_folder), str(filename)))
				values.append(pre[necpd_idx] if pre is not None else np.nan)
			df_new[method_name] = values

	to_plot = df_new[[methodX,methodY,'dataset','filename']]
	fig = px.box(to_plot[[methodX,methodY]],labels={
					 "value": "{}".format(measure),
					 "variable": "{}".format('methods'),
				 },title="Average {} on {} time series ({})".format(measure,dataset,anoma_type))
	fig.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=375)
	
	fig_scatter = px.scatter(to_plot,x=methodX, y=methodY,color='dataset',hover_name='filename',marginal_x='histogram', marginal_y='histogram')
	fig_scatter.update_traces(
		marker=dict(size=8,
		line=dict(width=1,
		color='DarkSlateGrey')),
		selector=dict(mode='markers'))
	fig_scatter.add_trace(go.Scatter(x=[0,1], y=[0,1],
		mode='lines',name='equality lines',line=dict(width=2,color='black'))
	)
	
	fig_scatter.update_yaxes(rangemode="tozero")
	fig_scatter.update_xaxes(rangemode="tozero")
	fig_scatter.update_layout(template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=675)
	return [dcc.Graph(figure=fig,id='boxplot_page_2')],[dcc.Graph(figure=fig_scatter,id='scatter_page_2')]


@app.callback(Output('ts_place_comp','children'),
	[Input('scatter_page_2','clickData'),
	Input('methodX_select_page_2', 'value'),
	Input('methodY_select_page_2', 'value'),])
def update_page2_timeseries_on_click(clickData,methodX,methodY):
	if clickData is None or methodX in (None, 'ALL') or methodY in (None, 'ALL'):
		return None

	filename = clickData['points'][0]["hovertext"]
	dataset_folder = df.loc[df['filename']==filename]['dataset'].values[0]
	folder, ts_name, ts_path, scores_dir = _resolve_paths(dataset_folder, filename)

	ts = pd.read_csv(ts_path + ".zip",compression='zip', header=None).to_numpy()
	label = ts[:,1]
	data = ts[:,0].astype(float)
	orig_len = len(data)
	data_min = float(np.min(data)) if orig_len else 0.0
	data_max = float(np.max(data)) if orig_len else 0.0

	plot_idx = None
	if orig_len > LIMIT_POINT_TS:
		base_idx = np.linspace(0, orig_len - 1, num=LIMIT_POINT_TS, dtype=int)
		anom_idx = np.where(label == 1)[0]
		extras = np.concatenate([anom_idx, anom_idx - 1, anom_idx + 1]) if len(anom_idx) else np.array([], dtype=int)
		plot_idx = np.unique(np.concatenate([base_idx, extras]))
		plot_idx = plot_idx[(plot_idx >= 0) & (plot_idx < orig_len)]
		label = label[plot_idx]
		data = data[plot_idx]
		x = plot_idx.tolist()
	else:
		x = list(range(orig_len))

	scores = {}
	for method_name in (methodX, methodY):
		if method_name in NECPD_METHODS:
			necpd_zip = _necpd_score_zip_path(method_name, folder, ts_name)
			if os.path.isfile(necpd_zip):
				scores_tmp = pd.read_csv(necpd_zip, compression="zip", header=None).to_numpy()
				score_vec = scores_tmp[:, 0].astype(float)
				if plot_idx is not None and len(score_vec) == orig_len:
					score_vec = score_vec[plot_idx]
				scores[method_name] = score_vec
			else:
				try:
					from src.models.normalized_entropy_cpd import NormalizedEntropyCPD

					window = max(2, min(int(NECPD_METHODS[method_name]), len(data)))
					clf = NormalizedEntropyCPD(window=window, bins="ln", score="delta")
					clf.fit(data)
					scores[method_name] = _normalize_scores(clf.decision_scores_)
				except Exception:
					pass
		elif method_name == ARIMA_METHOD:
			arima_zip = _necpd_score_zip_path(method_name, folder, ts_name)
			if os.path.isfile(arima_zip):
				scores_tmp = pd.read_csv(arima_zip, compression="zip", header=None).to_numpy()
				score_vec = scores_tmp[:, 0].astype(float)
				if plot_idx is not None and len(score_vec) == orig_len:
					score_vec = score_vec[plot_idx]
				scores[method_name] = score_vec
			elif len(data) <= 5000:
				try:
					from src.models.arima_cpd import ARIMACPD

					clf = ARIMACPD(order=ARIMA_ORDER)
					clf.fit(data)
					scores[method_name] = _normalize_scores(clf.decision_scores_)
				except Exception:
					pass
		else:
			score_zip = os.path.join(scores_dir, method_name, "score", ts_name) + ".zip"
			if os.path.isfile(score_zip):
				scores_tmp = pd.read_csv(score_zip,compression='zip', header=None).to_numpy()
				score_vec = scores_tmp[:,0].astype(float)
				if plot_idx is not None and len(score_vec) == orig_len:
					score_vec = score_vec[plot_idx]
				scores[method_name] = score_vec

	anom = add_rect(label,data)
	trace_scores = []
	trace_scores.append(go.Scattergl(
		x=x,
		y=data,
		xaxis='x',
		yaxis='y2',
		name = "Time series",
		mode = 'lines',
		line = dict(color = 'blue',width=3),
		opacity = 1
	))
	trace_scores.append(go.Scattergl(
		x=x,
		y=anom,
		xaxis='x',
		yaxis='y2',
		name = "Anomalies",
		mode = 'lines',
		line = dict(color = 'red',width=3),
		opacity = 1
	))

	for method_name in scores.keys():
		if len(scores[method_name]) != len(x):
			continue
		trace_scores.append(go.Scattergl(
			x=x,
			y=scores[method_name],
			name = "{} score".format(method_name),
			opacity = 1,
			mode = 'lines',
			fill="tozeroy",
		))

	layout = go.Layout(
		yaxis=dict(
			domain=[0, 0.4],
			range=[0,1]
		),
		yaxis2=dict(
			domain=[0.45, 1],
			range=[data_min, data_max]
		),
		title="{} time series snippet ({}k points maximum)".format(filename.split(".")[0], int(LIMIT_POINT_TS/1000)),
		template="simple_white",
		margin=dict(l=8, r=4, t=50, b=10),
		height=375,
		hovermode="x unified",
		xaxis=dict(
			range=[0,orig_len]
		)
	)

	fig = dict(data=trace_scores, layout=layout)
	return [dcc.Graph(figure=fig,id='ts_place_2',style={'width': '100%'})]



############################## page 3 ################################

@app.callback(
	[Output('title_table_3','children'),
	Output('res_table_3','children')], 
	[Input('dataset_select_page_3', 'value'),
	Input('exp_select_page_3', 'value'),
	Input('type_plot_page_3', 'value')
	])
def update_page3_robustness_summary(dataset,exp,plot_type):
	df_new = global_dataframe
	if dataset not in (None, 'ALL'):
		df_new = df_new.loc[df_new['folder'] == dataset]

	if exp in ('noise', 'lag', 'ratio'):
		df_new = df_new.loc[df_new['type'] == exp]

	if exp is None:
		exp = 'lag,noise, and ratio'
	if dataset is None:
		dataset = 'ALL'
	df_new = df_new.round(3)
	to_plot = df_new

	if plot_type is None:
		plot_type = 'boxplot'

	if plot_type == 'boxplot':
		fig = px.box(to_plot,y="value",x="measure",labels={
			"value": "{}".format("standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'mean':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').mean().sort_values('value',ascending=False),labels={
			"_value": "{}".format("average standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'median':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').median().sort_values('value',ascending=False),labels={
			"_value": "{}".format("median standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'min':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').min().sort_values('value',ascending=False),labels={
			"_value": "{}".format("minimal standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})
	elif plot_type == 'max':
		fig = px.bar(to_plot[['measure','value']].groupby('measure').max().sort_values('value',ascending=False),labels={
			"_value": "{}".format("maximal standard deviation"),
			"measure": "{}".format('Accuracy measures'),
		})

	fig.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=300)
	

	return html.H5("standard deviation when we inject {} in the anomaly score on {} time series".format(exp,dataset)),[dcc.Graph(figure=fig,id='box_place_3',style={'width': '100%'})]


def generate_new_label(label,lag):
	if lag < 0:
		return np.array(list(label[-lag:]) + [0]*(-lag))
	elif lag > 0:
		return np.array([0]*lag + list(label[:-lag]))
	return label

def generate_curve(grader,label,score,slidingWindow):
	_, _, _, _, avg_auc_3d, avg_ap_3d = grader.RangeAUC_volume(
		labels_original=label, score=score, windowSize=slidingWindow
	)
	return avg_auc_3d, avg_ap_3d

@app.callback(
	output=[
		Output('title_table_3_1','children'),
		Output('res_table_3_1','children'),
		Output('res_ts_3','children')
	], 
	inputs=[
		Input('time_series_select_page_3', 'value'),
		Input('exp_select_page_3_1', 'value'),
		Input('type_plot_page_3_1', 'value'),
		Input('method_select_page_3', 'value'),
		Input('condition_custom_page_3', 'value')
	],)
def update_graphs_page_measure(time_series,exp,plot_type,method,condition_custom):
	if (condition_custom is not None) and (time_series is not None) and (method is not None) and (exp is not None) and (plot_type is not None):
		dataset_folder = df.loc[df['filename']==time_series]['dataset'].values[0]
		folder, ts_name, ts_path, scores_dir = _resolve_paths(dataset_folder, time_series)

		ts = pd.read_csv(ts_path+ '.zip',compression='zip', header=None).to_numpy()
		label = ts[:,1]
		data = ts[:,0].astype(float)
		x = list(range(len(data)))

		slidingWindow = find_length(data)

		if method in NECPD_METHODS:
			necpd_zip = _necpd_score_zip_path(method, folder, ts_name)
			if os.path.isfile(necpd_zip):
				scores = pd.read_csv(necpd_zip, compression="zip", header=None).to_numpy()[:, 0].astype(float)
			else:
				from src.models.normalized_entropy_cpd import NormalizedEntropyCPD

				window = max(2, min(int(NECPD_METHODS[method]), len(data)))
				clf = NormalizedEntropyCPD(window=window, bins="ln", score="delta")
				clf.fit(data)
				scores = _normalize_scores(clf.decision_scores_)
		elif method == ARIMA_METHOD:
			arima_zip = _necpd_score_zip_path(method, folder, ts_name)
			if os.path.isfile(arima_zip):
				scores = pd.read_csv(arima_zip, compression="zip", header=None).to_numpy()[:, 0].astype(float)
			elif len(data) <= 5000:
				try:
					from src.models.arima_cpd import ARIMACPD

					clf = ARIMACPD(order=ARIMA_ORDER)
					clf.fit(data)
					scores = _normalize_scores(clf.decision_scores_)
				except Exception:
					return None, None, None
			else:
				return None, None, None
		else:
			score_zip = os.path.join(scores_dir, method, "score", ts_name) + ".zip"
			scores = pd.read_csv(score_zip,compression='zip', header=None).to_numpy()[:,0].astype(float)

		dict_acc = {
				'R_AUC_ROC':      {},
				'AUC_ROC':        {},
				'R_AUC_PR':       {},
				'AUC_PR':         {},
				'VUS_ROC':        {},
				'VUS_PR':         {},
				'Precision':      {},
				'Recall':         {},
				'F':              {},
				'Precision@k':    {},
				'Rprecision':     {},
				'Rrecall':        {},
				'RF':             {}}

		if exp == 'lag':
			lag_range = list(range(-slidingWindow//4,slidingWindow//4,5))
		elif exp == 'noise':	
			lag_range = [0.01,0.02,0.05,0.07,0.1,0.12,0.15,0.17,0.2]
		else:
			return None,None,None

		grader = metricor()
		scores_by_lag = {}

		for lag in lag_range:
			if exp == 'lag':
				new_label = generate_new_label(label,lag)
				new_scores = scores
			else:
				new_label = label
				noise = np.random.normal(-lag,lag,len(scores))
				new_scores = _normalize_scores(scores + noise)
				scores_by_lag[lag] = new_scores

			R_AUC, R_AP, *_ = grader.RangeAUC(labels=new_label, score=new_scores, window=slidingWindow, plot_ROC=True) 
			L = grader.metric_new(new_label, new_scores)
			_, _, AP = grader.metric_PR(new_label, new_scores)
			avg_auc_3d, avg_ap_3d = generate_curve(grader,new_label,new_scores,2*slidingWindow)

			dict_acc['R_AUC_ROC'][lag] 		=R_AUC
			dict_acc['AUC_ROC'][lag]        =L[0]
			dict_acc['R_AUC_PR'][lag]       =R_AP
			dict_acc['AUC_PR'][lag]         =AP
			dict_acc['VUS_ROC'][lag]        =avg_auc_3d
			dict_acc['VUS_PR'][lag]         =avg_ap_3d
			dict_acc['Precision'][lag]      =L[1]
			dict_acc['Recall'][lag]         =L[2]
			dict_acc['F'][lag]              =L[3]
			dict_acc['Precision@k'][lag]    =L[9]
			dict_acc['Rprecision'][lag]     =L[7]
			dict_acc['Rrecall'][lag]        =L[4]
			dict_acc['RF'][lag]             =L[8]

			
		##### stat plot

		dict_acc_df = pd.DataFrame(dict_acc)[pd.DataFrame(dict_acc).std().sort_values(ascending=False).index]
		fig_box = px.box(dict_acc_df,labels={
			"value": "{}".format("value"),"variable": "{}".format('Accuracy measures')})
		fig_box.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=230)
		fig_box_evo = px.line(dict_acc_df,markers=True,labels={
			"value": "{}".format("value"),"index": "{} injected".format(exp)})
		fig_box_evo.update_layout(showlegend=True,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=460,hovermode="x unified")
		fig_bar = px.bar(dict_acc_df.std(),labels={
			"value": "{}".format("standard deviation"),"index": "{}".format('Accuracy measures')})
		fig_bar.update_layout(showlegend=False,template="simple_white",margin=dict(l=8, r=4, t=50, b=10),height=230,)
		


		col_box = dbc.Row([
			dbc.Col([
					dbc.Row([
						dbc.Col([
							dcc.Graph(figure=fig_box,id='box_plot',style={'width': '100%'},config={'displayModeBar': False})
							],width=12),
					]),
					dbc.Row([
						dbc.Col([
							dcc.Graph(figure=fig_bar,id='bar_plot',style={'width': '100%'},config={'displayModeBar': False})
						],width=12),
					]),
				],width=6),
			dbc.Col(
				[dcc.Graph(figure=fig_box_evo,id='box_plot_evo',style={'width': '100%'},config={'displayModeBar': False})],width=6)
			])

		##### ts

		anom = add_rect(label,data)
		trace_scores = []
		trace_scores.append(go.Scattergl(
			x=x,
			y=data,
			xaxis='x',
			yaxis='y2',
			name = "Time series",
			mode = 'lines',
			line = dict(color = 'blue',width=3),
			opacity = 1
		))
		trace_scores.append(go.Scattergl(
			x=x,
			y=anom,
			xaxis='x',
			yaxis='y2',
			name = "Anomalies",
			mode = 'lines',
			line = dict(color = 'red',width=3),
			opacity = 1
		))

		
		if exp == 'lag':
			for i,lag in enumerate(lag_range):
				if (i == 0) or (i == len(lag_range)-1):
					trace_scores.append(go.Scattergl(
						x=x,
						y=[0]*abs(min(lag,0))+ [0] + list(scores[1+max(lag,0):-1-abs(min(lag,0))]) + [0] + [0]*max(lag,0),
						name = "{} with {} lag".format(method,lag),
						line = dict(color = 'black'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))
				else:
					trace_scores.append(go.Scattergl(
						x=x,
						y=[0]*abs(min(lag,0))+ [0] + list(scores[1+max(lag,0):-1-abs(min(lag,0))]) + [0] + [0]*max(lag,0),
						name = "{} with {} lag".format(method,lag),
						line = dict(color = 'rgba(26,150,65,0.1)'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))
		elif exp == 'noise':
			for i,lag in enumerate(lag_range):
				new_scores = scores_by_lag.get(lag)
				if new_scores is None:
					noise = np.random.normal(-lag,lag,len(scores))
					new_scores = _normalize_scores(scores + noise)
				if (i == 0) or (i == len(lag_range)-1):
					trace_scores.append(go.Scattergl(
						x=x,
						y=[0] + list(new_scores[1:-1]) + [0],
						name = "{} with {} noise".format(method,lag),
						line = dict(color = 'black'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))
				else:
					trace_scores.append(go.Scattergl(
						x=x,
						y=[0] + list(new_scores[1:-1]) + [0],
						name = "{} with {} noise".format(method,lag),
						line = dict(color = 'rgba(26,150,65,0.1)'),
						mode = 'lines',
						fill="tozeroy",
						fillcolor='rgba(26,150,65,0.1)'
					))

		layout = go.Layout(
			yaxis=dict(
				domain=[0, 0.4],
				range=[0,1]
			),
			yaxis2=dict(
				domain=[0.45, 1],
				range=[min(data),max(data)]
			),
			#showlegend=False,
			title="{} time series snippet (40k points maximum)".format(time_series),
			template="simple_white",
			margin=dict(l=8, r=4, t=50, b=10),
			height=375,
			hovermode="x unified",
			xaxis=dict(
				range=[0,len(data)]
			)
		)

		fig = dict(data=trace_scores, layout=layout)

		ts_plot = [dcc.Graph(figure=fig,id='ts_place_3',style={'width': '100%','height':'80%'})]

		##### title	
		title = html.H5("{} Experiment on {} time series with {} method".format(exp,time_series,method))


		return title,col_box, ts_plot
	return None,None,None

if __name__ == "__main__":
	app.run(debug=False)
