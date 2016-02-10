try:
	from urllib2 import urlopen, quote
except:
	from urllib.request import urlopen, quote

import json
import http.client as hc

# Source
PORT = 9876
HOST = "localhost"
PROTOCOL = "http"
DATA_EXT=''
LIVE_SOURCE=True
PREFIX=''

PRINT_URLS=True
PRINT_ALL=False

def initialise(port=9876,host='localhost',protocol='http',prefix='',live=True):
	global PORT, HOST, PROTOCOL, DATA_EXT, BASE_URL,LIVE_SOURCE
	PORT=port
	HOST=host
	PROTOCOL=protocol
	PREFIX=prefix
	BASE_URL = "%s://%s:%d%s" % (PROTOCOL,HOST,PORT,PREFIX)
	LIVE_SOURCE=live
	if LIVE_SOURCE:
		DATA_EXT=''
	else:
		DATA_EXT='.json'

def retrieve_resource(url,ext):
	if print_urls:
		print("*** %s ***" % (url))

	save_data(url[1:],urlopen(base_url+quote(url)).read(),ext,mode="b")

def retrieve_json(url):
	if PRINT_URLS:
		print("*** %s ***" % (url))

	text = urlopen(BASE_URL + quote(url+DATA_EXT)).read().decode('utf-8')
	
	if PRINT_ALL:
		print(json.loads(text))
		print("")
	return json.loads(text)

def update_json(url,data):
	conn = hc.HTTPConnection('localhost',port=9876)
	conn.request('PUT',url,json.dumps(data),headers={'Content-type':'application/json','Accept':'application/json'})
#	resp = conn.getresponse()
#	resp.code
	return conn

def run_model(params={}):
	conn = hc.HTTPConnection('localhost',port=9876)
#	conn.request('POST','/runs',json.dumps({'parameters':params}),headers={'Content-type':'application/json','Accept':'application/json'})
	conn.request('POST','/runs',json.dumps(params),headers={'Content-type':'application/json','Accept':'application/json'})
	resp = conn.getresponse()
	if resp.getcode()==302:
		return resp.getcode(),resp.getheader('Location')
	else:
		return resp.getcode(),None

def retrieve_run(run='latest'):
	if run=='latest' and not LIVE_SOURCE:
		all_runs = retrieve_json('/runs')
		return retrieve_json(all_runs[-1]['RunUrl'])

	return retrieve_json('/runs/%s'%str(run))

def result_matches_criteria(result,criteria):
	import re
	for key,pattern in criteria.items():
		if not re.match(pattern,result[key]):
			return False
	return True
	
def name_time_series(result):
	return result['TimeSeriesName']

def name_element_variable(result):
	element = result['NetworkElement']
	variable = result['RecordingVariable'].split(' - ')[-1]
	return '%s:%s'%(element,variable)

def retrieve_multiple_time_series(run='latest',run_data=None,criteria={},timestep='daily',name_fn=name_element_variable):
	"""
	Retrieve multiple time series from a run according to some criteria.

	Return all time series in a single Pandas DataFrame with date time index.

	Crtieria should be regexps for the fields in a Veneer time series record:
	  * RecordingElement
	  * RecordingVariable
	  * TimeSeriesName
	  * TimeSeriesUrl

	timestep should be one of 'daily' (default), 'monthly', 'annual'
	"""
	from pandas import DataFrame
	if timestep=="daily":
		suffix = ""
	else:
		suffix = "/aggregated/%s"%timestep

	if run_data is None:
		run_data = retrieve_run(run)

	retrieved={}

	for result in run_data['Results']:
		if result_matches_criteria(result,criteria):
			retrieved[name_fn(result)] = retrieve_json(result['TimeSeriesUrl']+suffix)['Events']

	if len(retrieved) == 0:
		return DataFrame()
	else:
		index = [event['Date'] for event in retrieved.values()[0]]
		data = {k:[event['Value'] for event in result] for k,result in retrieved.items()}
		return DataFrame(data=data,index=index)