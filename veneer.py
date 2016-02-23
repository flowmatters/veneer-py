try:
	from urllib2 import urlopen, quote
except:
	from urllib.request import urlopen, quote

import json
import http.client as hc

# Source

PRINT_URLS=True
PRINT_ALL=False

class Veneer(object):
	def __init__(self,port=9876,host='localhost',protocol='http',prefix='',live=True):
		self.port=port
		self.host=host
		self.protocol=protocol
		self.prefix=prefix
		self.base_url = "%s://%s:%d%s" % (protocol,host,port,prefix)
		self.live_source=live
		if self.live_source:
			self.data_ext=''
		else:
			self.data_ext='.json'
		self.model = VeneerIronPython(self)

#	def retrieve_resource(self,url,ext):
#		if PRINT_URLS:
#			print("*** %s ***" % (url))
#
#		save_data(url[1:],urlopen(base_url+quote(url)).read(),ext,mode="b")

	def retrieve_json(self,url):
		if PRINT_URLS:
			print("*** %s ***" % (url))

		text = urlopen(self.base_url + quote(url+self.data_ext)).read().decode('utf-8')
		
		if PRINT_ALL:
			print(json.loads(text))
			print("")
		return json.loads(text)

	def update_json(self,url,data,async=False):
		return self.send_json(url,data,'PUT',async)

	def send_json(self,url,data,method,async=False):
		conn = hc.HTTPConnection(self.host,port=self.port)
		payload = json.dumps(data)
		conn.request(method,url,payload,headers={'Content-type':'application/json','Accept':'application/json'})
		if async:
			return conn
		resp = conn.getresponse()
		code = resp.getcode()
		if code==302:
			return code,resp.getheader('Location')
		elif code==200:
			return code,json.loads(resp.read().decode('utf-8'))
		else:
			return code,None

		return conn

	def post_json(self,url,data,async=False):
		return self.send_json(url,data,'POST',async)

	def run_server_side_script(self,script,async=False):
		#print(script)
		result = self.post_json('/ironpython',{'Script':script},async=async)
		if async:
			return result
		code,data = result
		if code == 403:
			raise Exception('Script disabled. Enable scripting in Veneer')
		return data

	def run_model(self,params={},async=False):

		conn = hc.HTTPConnection(self.host,port=self.port)
	#	conn.request('POST','/runs',json.dumps({'parameters':params}),headers={'Content-type':'application/json','Accept':'application/json'})
		conn.request('POST','/runs',json.dumps(params),headers={'Content-type':'application/json','Accept':'application/json'})
		if async:
			return conn

		resp = conn.getresponse()
		if resp.getcode()==302:
			return resp.getcode(),resp.getheader('Location')
		else:
			return resp.getcode(),None

	def retrieve_run(self,run='latest'):
		if run=='latest' and not self.live_source:
			all_runs = self.retrieve_json('/runs')
			return self.retrieve_json(all_runs[-1]['RunUrl'])

		return self.retrieve_json('/runs/%s'%str(run))

	def result_matches_criteria(self,result,criteria):
		import re
		for key,pattern in criteria.items():
			if not re.match(pattern,result[key]):
				return False
		return True
		
	def name_time_series(self,result):
		return result['TimeSeriesName']

	def name_element_variable(self,result):
		element = result['NetworkElement']
		variable = result['RecordingVariable'].split(' - ')[-1]
		return '%s:%s'%(element,variable)

	def retrieve_multiple_time_series(self,run='latest',run_data=None,criteria={},timestep='daily',name_fn=name_element_variable):
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
			run_data = self.retrieve_run(run)

		retrieved={}

		for result in run_data['Results']:
			if self.result_matches_criteria(result,criteria):
				retrieved[name_fn(result)] = self.retrieve_json(result['TimeSeriesUrl']+suffix)['Events']

		if len(retrieved) == 0:
			return DataFrame()
		else:
			index = [event['Date'] for event in retrieved.values()[0]]
			data = {k:[event['Value'] for event in result] for k,result in retrieved.items()}
			return DataFrame(data=data,index=index)

class VeneerIronPython(object):
	def __init__(self,veneer):
		self._veneer = veneer

	def _initScript(self,namespace=None):
		script = "# Generated Script\n"
		if not namespace is None:
			script += "import %s\n"%namespace
		script += "import clr\n"
		script += "import System\n"
		script += "clr.ImportExtensions(System.Linq)\n"
		return script

	def runScript(self,script,async=False):
		return self._veneer.run_server_side_script(script,async)

	def sourceHelp(self,theThing='scenario',namespace=None):
		"""
		Get some help on what you can do with theThing,
		where theThing is something you can access from a scenario, or the scenario itself.

		If theThing is a method, returns details on how to call the method
		If theThing is an object, returns a list of public methods and properties

		eg
		v.model.sourceHelp() # returns help on the scenario object
		v.model.sourceHelp('scenario.CurrentConfiguration')
		"""
		theThing = theThing.replace('.*','.First().')

		script = self._initScript(namespace)
		script += "if hasattr(%s, '__call__'):\n"%theThing
		script += "    result = 'function'\n"
		script += "    help(%s)\n"%theThing
		script += "else:\n"
		script += "    result = dir(%s)"%theThing
		data = self.runScript(script)
		if not data['Exception'] is None:
			raise Exception(data['Exception'])
		if(data['Response']['Value']=='function'):
		    print(data['StandardOut'])
		else:
		    return [d['Value'] for d in data['Response']['Value']]

	def _generateLoop(self,theThing,innerLoop):
		script = ''
		indent = 0
		indentText = ''
		levels = theThing.split('.*')
		prevLoop = ''
		for level in levels[0:-1]:
			loopVar = "i_%d"%indent
			script += indentText+'for %s in %s%s:\n'%(loopVar,prevLoop,level)
			indent += 1
			indentText = ' '*(indent*4) 
			prevLoop = loopVar+'.'
		script += innerLoop.replace('<I>',indentText)%(prevLoop,levels[-1])
		script += '\n'
		return script

	def get(self,theThing,namespace=None):
		script = self._initScript(namespace)
		listQuery = theThing.find(".*") != -1
		if listQuery:
			script += 'result = []\n'
			innerLoop = '<I>result.append(%s%s)'
			script += self._generateLoop(theThing,innerLoop)
		else:
			script += "result = %s\n"%theThing
#		return script
		resp = self.runScript(script)
		if not resp['Exception'] is None:
			raise Exception(resp['Exception'])
		data = resp['Response']['Value']
		if listQuery:
			return [d['Value'] for d in data]
		return data

	def set(self,theThing,theValue,namespace=None,literal=True,fromList=False):
		if literal and isinstance(theValue,str):
			theValue = "'"+theValue+"'"
		script = self._initScript(namespace)
		script += 'origNewVal = %s\n'%theValue
		if fromList:
			script += 'origNewVal.reverse()\n'
			script += 'newVal = origNewVal[:]\n'
		else:
			script += 'newVal = origNewVal\n'

		innerLoop = "<I>%s%s = newVal"
		if fromList:
			innerLoop += '.pop()\n'
			innerLoop += '<I>if len(newVal)==0: newVal = origNewVal[:]\n'
		script += self._generateLoop(theThing,innerLoop)

#		return script
		result = self.runScript(script)
		if not result['Exception'] is None:
			raise Exception(result['Exception'])
		return None

	def sourceScenarioOptions(self,optionType,option=None,newVal = None):
		script = self._initScript('RiverSystem.ScenarioConfiguration.%s as %s'%(optionType,optionType))
		retrieve = "scenario.GetScenarioConfiguration[%s]()"%optionType
		if option is None:
			script += "result = dir(%s)\n"%retrieve
		else:
			if not newVal is None:
				script += "%s.%s = %s\n"%(retrieve,option,newVal)
			script += "result = %s.%s\n"%(retrieve,option)
		return self.runScript(script)
