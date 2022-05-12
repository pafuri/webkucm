#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datashader as ds
import pandas as pd
import colorcet as cc
import holoviews as hv
#import geoviews as gv
import geopandas as gpd
import plotly.graph_objs as go
from holoviews.element.tiles import EsriImagery
from holoviews.operation.datashader import datashade
import datashader.transfer_functions as tf
from colorcet import fire
import dash
from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate


import plotly.express as px
from shapely.geometry import Polygon,Point
import numpy as np
import h5py

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
#import dash
#from dash import html
#import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
from holoviews.operation.datashader import datashade
#import pandas as pd
#import numpy as np
#from plotly.data import carshare
from plotly.colors import sequential

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
def maskCirc(a):
    h, w = a.shape[:2]
    mask = create_circular_mask(h, w)
    masked_img = a.copy()
    masked_img[~mask] = np.nan
    return masked_img



mapbox_token = "pk.eyJ1IjoibWJleWVycyIsImEiOiJjampseXhhNDgwN3BhM2xvbjl4dWlzaXUxIn0.80ZkloVVWrPJi3r-rlnPZQ"

url = 'mapbox://styles/mbeyers/cl2yxpwzs001114o8szll7dpq' # navigationlight
url = 'mapbox://styles/mbeyers/cjrcnby6e2n9r2sqkukhmpyvu' # satellite

#da = np.random.normal(3, 2.5, size=(10,10,10))


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = Dash(__name__, external_stylesheets=external_stylesheets)
app = Dash(__name__)


fileName = "Results/ucmdata.hdf5"
hdf = h5py.File(fileName, 'r',libver='latest', swmr=True)
tseries =  np.array(hdf['Meta']['Time'])#,dtype='datetime64')
tseries = tseries.astype(dtype='datetime64').tolist()
metrics = list(hdf['Data'].keys())

Ta_df = np.array(hdf['Data']['Ta'])
Tdb_df = np.array(hdf['Data']['Tdb'])

gr_df = np.array(hdf['Data']['LAI'][:,:,0])
gr_df = maskCirc(gr_df[::-1,:])



Ta = np.array(hdf['Data']['Ta'][:,:,24*3+15])
Ta = maskCirc(Ta[::-1,:])

Ta_night = np.array(hdf['Data']['Ta'][:,:,24*3+21])
Ta_night = maskCirc(Ta_night[::-1,:])

Tmin = np.nanmean(Ta)-2*np.nanstd(Ta)
Tmax = np.nanmean(Ta)+2*np.nanstd(Ta)
print(Tmin,Tmax)
#Ta[np.isnan(Ta)]=25
#plt.imshow(Ta)
#plt.show()


metrics = list(hdf['Data'].keys())
#tseries = np.arange(len(tseries))

grid = np.indices(np.shape(Ta))
lat,lon = 39.001323930499245, -77.02511122482227
lat,lon, = 42.35, -71.058183
grid_lon = grid[1,:,:]*0.001071237279057250238+lon-374/2*0.001071237279057250238 
grid_lat = grid[0,:,:]*0.001071237279057250238+lat-374/2*0.001071237279057250238 

df = pd.DataFrame(grid_lat.flatten(),columns=['Lat'])
df['Lon']=grid_lon.flatten()
df['Values'] = Ta.flatten()
df['Values2'] = Ta_night.flatten()
df['x'] = grid[1,:,:].flatten()
df['y'] = 374-grid[0,:,:].flatten()


#gdf2 = gpd.GeoDataFrame(df.drop(['Lon', 'Lat'], axis=1),
#                                crs={'init': 'epsg:4326'},
#                                geometry=[Point(xy) for xy in zip(df.Lon, df.Lat)])
gdf2 = gpd.GeoDataFrame(df,crs={'init': 'epsg:4326'},geometry=[Point(xy) for xy in zip(df.Lon, df.Lat)])

#df["easting"],df["northing"] = hv.Tiles.lon_lat_to_easting_northing(df["Lon"], df["Lat"])
print(df.head())

df = df.dropna()




colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div([
	html.Div([
    dcc.Dropdown(
        id="resolution",
        options=[
            {'label': '10 m', 'value': 20},
            {'label': '20 m', 'value': 40},
            {'label': '30 m', 'value': 60},
            {'label': '40 m', 'value': 80},
            {'label': '50 m', 'value': 100},
            {'label': '60 m', 'value': 120},
            {'label': '70 m', 'value': 140},],
        value=20,
        clearable=False),

    dcc.Dropdown(
#                df['Indicator Name'].unique(),
#                hdf['Data'].keys(),
                metrics,'Ta',
#                'Fertility rate, total (births per woman)',
                id='crossfilter-xaxis-column',
            ),    
    dcc.RangeSlider(
                id='condition-range-slider',
                min=0,
                max=50,
                value=[Tmin, Tmax],
                allowCross=False)],
    style={'width': '49%', 'display': 'inline-block'}),


	html.Div([
            dcc.Dropdown(
#                df['Indicator Name'].unique(),
				metrics,'Tdb',
                id='crossfilter-yaxis-column'
            ),
            dcc.RadioItems(
                ['Hex', 'Points'],
                'Hex',
                id='plot_type',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),
            dcc.RadioItems(
                ['Day', 'Night'],
                'Day',
                id='day_night',
                labelStyle={'display': 'inline-block', 'marginTop': '5px'}
            ),

            dcc.Slider(
                id='time-range-slider',
                min=0,
                max=24,
                value=15)
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

    html.Div([
    	dcc.Graph(id='hex',clickData={'points': [{'x': 165, 'y': 166, 'location': '-1.3433172229419912,0.7396997264742174','customdata': [35.4803458342493, '-1.3433172229419912,0.7396997264742174', 0]}]})], 
    	style={'width': '70%', 'display': 'inline-block', 'padding': '0 20'}),
	html.Div([
        dcc.Graph(id='x-time-series',
        	figure={
        	'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }}),
        dcc.Graph(id='y-time-series')],
#        dcc.Graph(id='crossfilter-indicator-scatter')],
        style={'display': 'inline-block', 'width': '30%'})
	])





@app.callback(
    Output('hex', 'figure'),
    Input('hex', 'clickData'),
    Input('hex', 'relayoutData'),
    Input('condition-range-slider', 'value'),
    Input('resolution', 'value'),
	Input('plot_type','value'),
	Input('day_night','value'))

#def update_graph3(resolution,condition_range,mapbox_cfg,clickData):
def update_graph3(clickData,mapbox_cfg,condition_range,resolution,plot_type,plot_time):
#	print(plot_time)

	ctx = dash.callback_context
#	print(ctx.triggered)
	selectevent = (ctx.triggered[0]['prop_id'].split('.')[0])
#	print(selectevent)
#	print(mapbox_cfg)

#	if mapbox_cfg and "mapbox.zoom" in mapbox_cfg.keys():
#		ll = mapbox_cfg['mapbox.center']
#		zoomi = mapbox_cfg['mapbox.zoom']
#		range_c=[condition_range[0],condition_range[1]]
#		reso = resolution
#	else:
#		ll = dict(lon=lon, lat=lat)
#		zoomi=12
#		range_c=[30,40]
#		reso = 30
	ll = dict(lon=lon, lat=lat)
	zoomi=12
#	range_c=[30,40]
#	reso = 30
	range_c=[condition_range[0],condition_range[1]]
	reso = resolution

	if (plot_time=="Day"):
		color_sel = "Values"
	if (plot_time=="Night"):
		color_sel = "Values2"

#	print(zoomi)
	flag = plot_type
	if (flag=="Hex"):
#		fig = ff.create_hexbin_mapbox(data_frame=df, lat="Lat", lon="Lon",color='Values',color_continuous_scale='RdYlBu_r',range_color=(range_c[0],range_c[1]),agg_func=np.mean,nx_hexagon=reso, opacity=0.6,zoom=zoomi,original_data_marker=dict(size=8, opacity=0.5, color=df['Values'],colorscale = 'RdYlBu_r'),show_original_data=False,center= {"lon": ll['lon'], "lat": ll['lat']})
		fig = ff.create_hexbin_mapbox(data_frame=df, lat="Lat", lon="Lon",color=color_sel,color_continuous_scale='RdYlBu_r',range_color=(range_c[0],range_c[1]),agg_func=np.mean,nx_hexagon=reso, opacity=0.6,zoom=zoomi,original_data_marker=dict(size=8, opacity=0.5, color=df['Values'],colorscale = 'RdYlBu_r'),show_original_data=False,center= {"lon": ll['lon'], "lat": ll['lat']})


#		fig.update_traces(marker_line_width=0,marker_line_color="#ffffff")

	if (flag=="Points"):
		fig = go.Figure(go.Scattermapbox(
    	lat= df['Lat'],
    	lon= df['Lon'],
        customdata = df[['Values','x','y']],
        mode='markers',
        marker=go.scattermapbox.Marker(
        	size= 10,
            colorscale = 'RdYlBu_r',
            color=df['Values'],
            cmin=range_c[0],
            cmax=range_c[1],
            opacity = .85,
            colorbar=dict(
                title='Temperature',
                thickness=20,
                titleside='right',
                outlinecolor='rgba(68,68,68,0)',
                ticks='outside',
                ticklen=3)) 
        )
    )
	


	fig.update_layout(margin=dict(b=0, t=0, l=0, r=0))
	fig.update_layout(autosize=False,uirevision=20,
		mapbox= dict(accesstoken=mapbox_token,
			zoom=zoomi,
			center= ll,
			style=url),
#			width=400,
#			height=400,
			title = "UHI")
#	fig.update_geos(projection_type="equirectangular")
#	fig.update_layout(mapbox_style="carto-darkmatter")
#	fig.update_layout(mapbox_style="basic")
#    fig.update_layout(mapbox_style=url)#    fig.update_layout(mapbox_style=url)




#	if (selectevent!='resolution'):
#		clickC = (clickData['points'][0]['customdata']) # 2nd entry in clickC from custom data is the id of the hex poluygon
#		gf = fig.data[0]
#		gdf=gpd.GeoDataFrame.from_dict(gf['geojson']['features'])#, orient='columns')
#		clickpoly = gdf[gdf['id']==clickC[1]]['geometry']
#		geom = clickpoly.iloc[0]
#		clickpoly = Polygon([tuple(l) for l in geom['coordinates'][0]])
#		ii = gdf2.within(clickpoly)
#		sgdf2 = gdf2.loc[ii]
#		pointsArr = sgdf2[['x','y']].values
#		print(pointsArr)
#		print("ind",pointsArr[0])

	if (selectevent=="hexs"):
		raise PreventUpdate
	else:
		return fig


def create_time_series(index,ilist):
#	print('full',ilist[:])
#	print(ilist[0:2].T[0])
#	print("start")
	xi = ilist[:].T[0]
	yi = ilist[:].T[1]
#	print(np.ix_(ilist))
#	arra = hdf['Data']['Ta'][:,:,:]
	arra = Ta_df



	t = arra[yi,xi,:]
	tm = np.average(t,axis=0)
	tmax = np.amax(t,axis=0)
	tmin = np.amin(t,axis=0)
#	print("v")

#	print(np.shape(t))
#	print(tseries)
##	plt.plot(t)
#	plt.show()

#	arr = arr.T
#	print(np.shape(arr))
#	arr2 = np.split(arr,239)
#	print(np.shape(arr2))
#	print("Index",index)
#	index=[50,50]
#	y1 = np.array(hdf['Data']['Ta'])[50,50,:]
#	y1 = np.array(hdf['Data']['Tdb'])[50,50,:]

	df = pd.DataFrame()
#	df['Ta'] = hdf['Data']['Ta'][index[1],index[0],:]

#	df['Tdb'] = hdf['Data']['Tdb'][index[1],index[0],:]
	df['Tdb'] = Tdb_df[index[1],index[0],:]

	df['Ta'] = t[0]
	df['Tmean'] = tm
	df['Tmax'] = tmax
	df['Tmin'] = tmin
	df['time'] = tseries
	df['date'] = pd.to_datetime(df['time'])#.dt.to_period('M')
#	print(df.head())

#	df['Ta2'] = hdf['Data']['Ta'][index[1]+10,index[0]+10,:]


#	df['Ta'] = hdf['Data']['Ta'][index['y'],index['x'],:]
#	df['Tdb'] = hdf['Data']['Tdb'][index['y'],index['x'],:]

#	fig = px.line(x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=np.array(hdf['Data']['Ta'])[50,50,:])
#	fig = px.line(x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=hdf['Data']['Ta'][50,50,:])
#	fig = px.line(df, x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=['Tdb','Tmean','Tmax','Tmin'])



#	fig = px.area(df, x='date',y=['Tmax','Tmin'])
#	fig = px.line(df, x='date',y=['Tdb','Tmean','Tmax','Tmin'])
#	fig = px.line(df, x='date',y=['Tdb','Tmean'])
	fig = go.Figure()
	fig.add_traces(go.Scatter(x=df['date'], y = df['Tmin'],line = dict(color='rgba(0,255,0,0)'),name='Tmin'))   
	fig.add_traces(go.Scatter(x=df['date'], y = df['Tmax'],mode='none',fill='tonexty',line_color="red",name='Tmax'))
	fig.add_traces(go.Scatter(x=df['date'], y = df['Tdb'],line_color="blue",name='Tdb'))
	fig.add_traces(go.Scatter(x=df['date'], y = df['Tmean'],line_color="black",name='Tmean'))
#	fig = px.line(df, x='date',y=['Tdb','Tmean'])

#	fig = px.add_trace(x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=hdf['Data']['Tdb'][50,50,:])
#	fig.update_traces(mode='lines+markers')
	fig.update_traces(mode='lines')
	fig.update_xaxes(showgrid=False,tickformat='%m-%d-%H<br>%Y')
#	fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
#	fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
#                       xref='paper', yref='paper', showarrow=False, align='left',
#                       text="tt")
	fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
#	print("d")

	return fig

def create_pie(index,ilist):
#	print('full',ilist[:])
#	print(ilist[0:2].T[0])
#	print("start")
	xi = ilist[:].T[0]
	yi = ilist[:].T[1]
#	print(np.ix_(ilist))
#	arra = hdf['Data']['Ta'][:,:,:]
	arra = gr_df



	t = arra[yi,xi]


#	print(t)
#	tm = np.average(t,axis=0)
#	tmax = np.amax(t,axis=0)
#	tmin = np.amin(t,axis=0)
#	print("v")

#	print(np.shape(t))
#	print(tseries)
##	plt.plot(t)
#	plt.show()

#	arr = arr.T
#	print(np.shape(arr))
#	arr2 = np.split(arr,239)
#	print(np.shape(arr2))
#	print("Index",index)
#	index=[50,50]
#	y1 = np.array(hdf['Data']['Ta'])[50,50,:]
#	y1 = np.array(hdf['Data']['Tdb'])[50,50,:]

	df = pd.DataFrame()
#	df['Ta'] = hdf['Data']['Ta'][index[1],index[0],:]

#	df['Tdb'] = hdf['Data']['Tdb'][index[1],index[0],:]
#	df['Tdb'] = Tdb_df[index[1],index[0],:]
	df['fgr'] = t
#	df['Tmean'] = tm
#	df['Tmax'] = tmax
#	df['Tmin'] = tmin
#	df['time'] = tseries

#	df['date'] = pd.to_datetime(df['time'])#.dt.to_period('M')
#	print(df.head())

#	df['Ta2'] = hdf['Data']['Ta'][index[1]+10,index[0]+10,:]


#	df['Ta'] = hdf['Data']['Ta'][index['y'],index['x'],:]
#	df['Tdb'] = hdf['Data']['Tdb'][index['y'],index['x'],:]

#	fig = px.line(x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=np.array(hdf['Data']['Ta'])[50,50,:])
#	fig = px.line(x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=hdf['Data']['Ta'][50,50,:])
#	fig = px.line(df, x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=['Tdb','Tmean','Tmax','Tmin'])
#	fig = px.line(df, x='date',y=['Tdb','Tmean','Tmax','Tmin'])

	fig = px.histogram(t, range_x=[0, 5],range_y=[0, 100],nbins=20,histnorm='percent')


#	fig = px.add_trace(x=np.arange(len(np.array(hdf['Data']['Ta'])[50,50,:])),y=hdf['Data']['Tdb'][50,50,:])
#	fig.update_traces(mode='lines+markers')
#	fig.update_traces(mode='lines')

#	fig.update_xaxes(showgrid=False,tickformat='%m-%d-%H<br>%Y')

#	fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
#	fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
#                       xref='paper', yref='paper', showarrow=False, align='left',
#                       text="tt")
	fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
#	print("d")

	return fig


@app.callback(
    Output('x-time-series', 'figure'),
    Output('y-time-series', 'figure'),
    Input('hex', 'clickData'),
    Input('hex', 'figure'),
    Input('resolution', 'value'))    
#    Input('hex','n_clicks'))
#    Input('crossfilter-yaxis-column', 'value'),
#    Input('crossfilter-yaxis-type', 'value'))

#def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
def update_timeseries(clickData,figo,reso):

#	print(relay)
#	print(clickData)
#	print(figo['data'][0]['geojson'])
#	print("tes")

	ctx = dash.callback_context
	selectevent = (ctx.triggered[0]['prop_id'].split('.')[0])
#	print(selectevent)

	if (selectevent!='resolution'):
#	print(clickData)
#		print("a")
		clickC = (clickData['points'][0]['customdata']) # 2nd entry in clickC from custom data is the id of the hex poluygon


		gf = figo['data'][0]#['geojson']
#	print(gf)
#	print('feat',gf['geojson']['features'])
		gdf=gpd.GeoDataFrame.from_dict(gf['geojson']['features'])#, orient='columns')
#	print(gdf.head())

#	name_by_id = dict([(str(gf['id']), p['name']) for p in data])

		clickpoly = gdf[gdf['id']==clickC[1]]['geometry']
#	print("cc",clickpoly)
		if clickpoly.empty:
#			print("empty")
			pointsArr=np.array([[152,186],[153,186]])
#			print("list",pointsArr)

		else:

			geom = clickpoly.iloc[0]
#	print(geom)
#	print(geom['coordinates'])
			clickpoly = Polygon([tuple(l) for l in geom['coordinates'][0]])

#	print(gdf2.head())

#	print(clickpoly)
#			print("b")
#			pointsArr=np.array([[50,50],[51,51]])
			ii = gdf2.within(clickpoly)
			sgdf2 = gdf2.loc[ii]
			pointsArr = sgdf2[['x','y']].values
#			print("c")

	else:
		pointsArr=np.array([[152,186],[153,186]])
#	print(relay)
#	index = clickData['points'][0]
#	d = (clickData['points'][0]['customdata'])
#	index = d[1:]
	index = pointsArr[0]
	return create_time_series(index,pointsArr),create_pie(index,pointsArr)




if __name__ == '__main__':
	app.run_server(debug=True)

