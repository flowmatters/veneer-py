import string
import pandas as pd
import numpy as np

MODEL_RUN_SCRIPT_PREFIX='''
from System import Array
import TIME.DataTypes.TimeStep as TimeStep
import TIME.DataTypes.TimeSeries as TimeSeries
import System.DateTime as DateTime
import ${namespace}.${klass} as ${klass}
import TIME.Tools.ModelRunner as ModelRunner
model = ${klass}()

runner = ModelRunner(model)
'''

def _literal_timeseries(series,name):
    start = series.index[0]
    values = list(np.array(series))
    
    #  TimeSeries( DateTime startTime, TimeStep timeStep, double[] values )
    scriptlet = 'values_%s = Array[float](%s)\n'%(name,values)
    scriptlet += 'start_%s = DateTime(%d,%d,%d)\n'%(name,start.year,start.month,start.day)
    scriptlet += 'timestep_%s = TimeStep.Daily\n'%name
    scriptlet += 'ts_%s = TimeSeries(start_%s,timestep_%s,values_%s)\n\n'%(name,name,name,name)
    return scriptlet

def _play(series,name):
    scriptlet = _literal_timeseries(series,name)
    scriptlet += 'runner.Play("%s",ts_%s)\n'%(name,name)
    return scriptlet

def _record(output):
    return 'runner.Record("%s")\n'%(output)

def _retrieve(output):
    return 'result["%s"] = runner.GetRecorded("%s")\n'%(output,output)

class VeneerComponentModelActions(object):
    def __init__(self,ironpy):
        self._ironpy = ironpy

    def run_model(self,model_name,namespace=None,inputs={},parameters={},outputs=[],extra_initialisation=''):
        '''
        Run the given component model with specified inputs, parameters and recorded outputs.

        Return:

        Dataframe of output time series
        '''
        if namespace is None:
            name_elements = model_name.split('.')
            namespace = '.'.join(name_elements[:-1])
            model_name = name_elements[-1]       

        tmpl = string.Template(MODEL_RUN_SCRIPT_PREFIX)

        run_script = tmpl.substitute(namespace=namespace,klass=model_name)
        run_script += extra_initialisation

        for k,ts in inputs.items():
            run_script += _play(ts,k)

        for k,val in parameters.items():
            run_script += 'model.%s = %s\n'%(k,str(val))

        run_script += ''.join([_record(o) for o in outputs])

        run_script += 'runner.execute()\n'
        run_script += 'result = {}\n'
        run_script += ''.join([_retrieve(o) for o in outputs])
        res = self._run_model(run_script)
        return self._transform_results(res)

    def _run_model(self,script):
        res = self._ironpy._safe_run(script)
        if res['Exception'] is not None:
            raise Exception(res['Exception'])
        entries = res['Response']['Entries']
        return {e['Key']['Value']:e['Value'] for e in entries}

    def _transform_results(self,res):
        index = [self._ironpy._veneer.parse_veneer_date(e['Date']) for e in list(res.values())[0]['Events']]
        data = {k:[e['Value'] for e in vals['Events']] for k,vals in res.items()}
        return pd.DataFrame(data,index=index)
