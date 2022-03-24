import io
import json
import os
import tempfile
from typing import DefaultDict
import pandas as pd
from veneer.actions import get_big_data_source
import veneer
from string import Template
import argparse
import logging

from veneer.manage import VENEER_EXE_FN, create_command_line, kill_all_now
import veneer.manage as manage
logger = logging.getLogger(__name__)

def _BEFORE_BATCH_NOP(slf,x,y):
    pass

STANDARD_SOURCE_ELEMENTS=[
    'Downstream Flow Volume',
    'Upstream Flow Volume'
]

STANDARD_SOURCE_VARIABLES=[
    'Quick Flow',
    'Slow Flow',
    'Total Flow'
]
SOURCE_STORAGE_VARIABLES=[
    'Storage Volume',
    'Water Surface Area',
    'Water Surface Elevation',
    'Regulated Release Volume',
    'Spill Volume',
    'Total Inflow Volume',
    'Total Outflow Volume'
]

CONSTITUENT_VARIABLES=[
    'Downstream Flow Mass',
    'Total Flow Mass',
    'Stored Concentration',
    'Stored Mass'
]

CONSTITUENT_VARIABLE_LABELS=[
    'network',
    'generation',
    'storage_concentration',
    'storage_mass'
]

CONSTITUENT_NAME_FN=[
    veneer.name_for_location,
    veneer.name_for_fu_and_sc,
    veneer.name_for_location,
    veneer.name_for_location
]

class SourceExtractor(object):
    def __init__(self,v,dest,results=None,climate_data_sources=['Climate Data'],progress=logger.info):
        self.v=v
        self.dest=dest
        self.current_dest=None
        self.results = results
        if self.results is None:
            self.results = os.path.join(self.dest,'Results')
        self.climate_data_sources = climate_data_sources
        self.progress=progress

    def _ensure(self):
        if not os.path.exists(self.current_dest):
            os.makedirs(self.current_dest)

    def writeable(self,fn):
        self._ensure()
        return open(os.path.join(self.current_dest,fn),'w')

    def write_json(self,fn,data):
        json.dump(data,self.writeable(fn+'.json'))

    def write_csv(self,fn,df):
        self._ensure()
        fn = os.path.join(self.current_dest,fn+'.csv')+'.gz'
        df.to_csv(fn,compression='gzip')

    def _extract_structure(self):
        fus = set(self.v.model.catchment.get_functional_unit_types())
        constituents = self.v.model.get_constituents()
        constituent_sources = self.v.model.get_constituent_sources()
        assert len(constituent_sources)==1 # We don't use constituent source

        network = self.v.network()
        network_df = network.as_dataframe()

        fu_areas = self.v.retrieve_csv('/tables/fus')
        fu_areas = pd.read_csv(io.StringIO(fu_areas),index_col=0)

        self.write_json('constituents',constituents)
        self.write_json('fus',list(fus))
        self.writeable('network.json').write(network_df.to_json())
        self.write_csv('fu_areas',fu_areas)

    def _retrieve_input_timeseries(self,input_table,name_columns=['NetworkElement'],name_join=':'):
        result = pd.DataFrame()
        cols = set(input_table.columns) - set(name_columns)

        for _,row in input_table.iterrows():
            row_name = name_join.join([row[nc] for nc in name_columns])
            for col in cols:
                if pd.isna(row[col]) or not row[col]:
                    continue
                path = row[col]
                name = f'{row_name} {col}'
                path_components = path.split('/')
                data_source = path_components[2]
                input_set = path_components[3]
                item = path_components[4]
                time_series = self.v.data_source_item(data_source,item,input_set=input_set)
                result[name] = time_series[time_series.columns[0]]
        return result

    def _extract_runoff_configuration(self):
        self._extract_models_and_parameters('catchment.runoff','runoff_models','rr')

        self.progress('Getting climate data')
        climate = pd.DataFrame()
        for ds_name in self.climate_data_sources:
            ds = get_big_data_source(self.v,ds_name,self.data_sources,self.progress)
            climate = pd.concat([climate,ds],axis=1)

        self.write_csv('climate',climate)

    def _extract_models_and_parameters(self,path,model_fn,param_prefix,**kwargs):
        source = self.v.model
        for p in path.split('.'):
            source = getattr(source,p)
        models = source.model_table(**kwargs)
        params = source.tabulate_parameters(**kwargs)
        self.write_csv(model_fn,models)
        for model_type, table in params.items():
            self.write_csv('%s-%s'%(param_prefix,model_type),table)

    def _extract_piecewise_routing_tables(self):
        piecewise_links = self.v.model.link.routing.get_param_values('link.Name if not i_0.FlowRouting.IsGeneric else None')
        piecewise_links = [p for p in piecewise_links if p is not None]
        res = {}
        for pl in piecewise_links:
            flows = self.v.model.link.routing.get_param_values('Piecewises.*IndexFlow',links=pl)
            travel_times = self.v.model.link.routing.get_param_values('Piecewises.*TravelTime',links=pl)
            df = pd.DataFrame({'IndexFlow':flows,'TravelTime':travel_times})
            res[pl] = df
            if not len(df):
                self.progress(f'Expected piecewise routing table for link {pl}, but no rows')
                continue
            self.write_csv('piecewise-routing-%s'%(pl),df)

    def _extract_routing_configuration(self):
        self._extract_models_and_parameters('link.routing','routing_models','fr')
        self._extract_models_and_parameters('link.constituents','transportmodels','cr')

        self._extract_piecewise_routing_tables()
        # link_models = self.v.model.link.routing.model_table()

        # transport_models = self.v.model.link.constituents.model_table()
        # transport_params = self.v.model.link.constituents.tabulate_parameters()
        # self.write_csv('routing_models',link_models)
        # for model_type, table in transport_params.items():
        #     self.write_csv('fr-%s'%model_type,table)

        # # self.write_csv('routing_params',link_params)
        # self.write_csv('transportmodels',transport_models)

        # for model_type, table in transport_params.items():
        #     self.write_csv('cr-%s'%model_type,table)

    def _extract_generation_configuration(self):
        self._extract_models_and_parameters('catchment.generation','cgmodels','cg')
        # generation_models = self.v.model.catchment.generation.model_table()
        # generation_parameters = self.v.model.catchment.generation.tabulate_parameters()

        # for model_type, table in generation_parameters.items():
        #     self.write_csv('cg-%s'%model_type,table)

        # self.write_csv('cgmodels',generation_models)

    def _extract_demand_configuration(self):
        extraction_params = self.v.model.node.tabulate_parameters(node_types='ExtractionNodeModel')
        
        if not len(extraction_params):
            self.progress('No extraction points. Skipping water user extraction')
            return

        extraction_params = extraction_params['RiverSystem.Nodes.SupplyPoint.ExtractionNodeModel']
        # extractions = list(extraction_params['NetworkElement'])
        self.write_csv('extraction_point_params',extraction_params)

        water_users = self.v.model.node.water_users.names()
        if len(water_users):
            self.progress('Extracting information for %d water users'%len(water_users))

        demands={}
        for wu in water_users:
            # d = self.v.model.node.water_users.get_param_values('DemandModel.Name',nodes=wu)[0]
            demand_type = self.v.model.node.water_users.get_param_values('DemandModel',nodes=wu)[0]
        #     assert len(d) == 1
        #     d = d[0]
            if demand_type.endswith('TimeSeriesDemandNodeModel'):
                demands[wu] = self.v.model.node.water_users.get_data_sources('DemandModel.Order',nodes=wu)[0]
            elif demand_type.endswith('MonthlyDemandNodeModel'):
                txt = self.v.model.node.water_users.get_param_values('DemandModel.Quantities',nodes=wu)[0]       
                demands[wu] = pd.DataFrame([{'month':ln[0],'volume':float(ln[1])} for ln in [ln.split(' ') for ln in txt.splitlines()]])
            else:
                raise Exception('Unsupported demand model: %s'%demand_type)

        for node,demand in demands.items():
            if isinstance(demand,pd.DataFrame):
                self.write_csv('monthly-pattern-demand-%s'%node,demand)
                continue

            if demand == '':
                logger.info('No demand time series configured for node: %s'%node)
                continue

            data_source = '/'.join(demand.split('/')[2:-2])
            ds = self.v.data_source(data_source)['Items'][0]['Details']
            ds = ensure_units(ds,'m3/s',lbl=data_source)

            self.write_csv('timeseries-demand-%s'%node,ds)

    def _extract_loss_configuration(self):
        loss_nodes = self.v.model.node.losses.names()
        if len(loss_nodes):
            self.progress('Extracting information for %d loss nodes'%len(loss_nodes))

        for loss in loss_nodes:
            loss_table = self.v.model.node.losses.loss_table(loss)
            self.write_csv('loss-table-%s'%loss,loss_table)

    def _extract_storage_configuration(self):
        params = self.v.model.node.storages.tabulate_parameters()

        if not len(params):
            logger.info('No storages in model')
            return

        params = params['RiverSystem.Nodes.StorageNodeModel']
        self.progress('Extracting information for %d storages'%len(params))

        self.write_csv('storage_params',params)

        for storage in list(params['NetworkElement']):
            self.progress('Extracting information for storage %s'%storage)
            lva = self.v.model.node.storages.lva(storage)
            self.write_csv('storage_lva_%s'%storage,lva)

        outlet_links = {}
        outlets={}
        releases={}
        for ne in list(params['NetworkElement']):
            outlet_links[ne] = self.v.model.node.storages.outlets(ne)
            for link in outlet_links[ne]:
                outlets[link] = self.v.model.node.storages.releases(ne,link)
                for rel in outlets[link]:
                    releases[(ne,rel)] = self.v.model.node.storages.release_table(ne,rel)
        storage_meta = {
            'outlet_links':outlet_links,
            'outlets':outlets
        }
        self.write_json('storage_meta',storage_meta)

        for (storage,release),table in releases.items():
            self.write_csv('storage_release_%s_%s'%(storage,release),table)

        self.progress('Extracting storage water quality configuration')
        self._extract_models_and_parameters('node.constituents','storage-wq','swq',nodes=list(params.NetworkElement),aspect='model')

        self.progress('Extracting storage climate data')
        input_map =  self.v.model.node.storages.tabulate_inputs()
        if len(input_map):
            input_map = input_map['RiverSystem.Nodes.StorageNodeModel']
        storage_climate = self._retrieve_input_timeseries(input_map)
        self.write_csv('storage_climate',storage_climate)

    def _write_data_source_timeseries(self,data_source_map,ref_col,fn_template):
        fn_template = Template(fn_template)
        SEC_TO_DAY=24*60*60
        ML_TO_M3 = 1e3
        ML_PER_DAY_TO_M3_PER_SEC=ML_TO_M3/SEC_TO_DAY
        MG_TO_KG=1e-6
        L_TO_M3=1e-3
        MG_PER_LITER_TO_KG_PER_M3 = MG_TO_KG/L_TO_M3
        for _,row in data_source_map.iterrows():
            data_source_path = row[ref_col]
            if (data_source_path is None) or not data_source_path.startswith('/dataSources/'):
                continue
                
            comp = data_source_path.split('/')
            data_source = comp[2]
            input_set = comp[3]
            column = comp[4]

            df = self.v.data_source_item(data_source,column,input_set)

            for col in df.columns:
                if not hasattr(df[col],'units'):
                    logger.debug('No units on ',col)
                    continue
                units = df[col].units
                if units=='ML/d':
                    df[col] *= ML_PER_DAY_TO_M3_PER_SEC
                elif units=='mg/L':
                    df[col] *= MG_PER_LITER_TO_KG_PER_M3
                df[col].units = units

            self.write_csv(fn_template.substitute(row),df)

    def _extract_external_inflows(self):
        inflow_params = self.v.model.node.tabulate_parameters(node_types='InjectedFlow')

        if not len(inflow_params):
            self.progress('No inflow nodes in model')
            return

        inflow_params = inflow_params['RiverSystem.Nodes.Inflow.InjectedFlow']
        inflow_data_sources = self.v.model.node.tabulate_inputs('InjectedFlow')
        self._write_data_source_timeseries(inflow_data_sources,'Flow','timeseries-inflow-${NetworkElement}')

        played_constituents = self.v.model.node.constituents.tabulate_inputs(node_types='InjectedFlow',aspect='played')
        played_constituents = played_constituents.get('RiverSystem.Constituents.ConstituentPlayedValue',None)

        if played_constituents is not None:
            self._write_data_source_timeseries(played_constituents,
                                            'ConstituentConcentration',
                                            'timeseries-inflow-concentration-${Constituent}-${NetworkElement}')

            self._write_data_source_timeseries(played_constituents,
                                            'ConstituentLoad',
                                            'timeseries-inflow-load-${Constituent}-${NetworkElement}')

    def extract_source_config(self):
        self.current_dest = self.dest
        self._ensure()

        self.progress('Getting data sources')
        self.data_sources = self.v.data_sources()

        self._extract_structure()
        self._extract_storage_configuration()
        self._extract_demand_configuration()
        self._extract_external_inflows()
        self._extract_loss_configuration()

        self._extract_runoff_configuration()
        self._extract_generation_configuration()
        self._extract_routing_configuration()

    def _get_recorder_batches(self):
        recorders = []
        for sse in STANDARD_SOURCE_ELEMENTS:
            recorders.append([{'recorder':{'RecordingElement':sse},'retriever':{'RecordingVariable':sse},'label':sse.replace(' ','_').lower()}])
        for ssv in STANDARD_SOURCE_VARIABLES:
            recorders.append([{'recorder':{'RecordingVariable':ssv},'label':ssv.replace(' ','_')}])
        recorders.append([{'recorder':{'RecordingVariable':sv},'label':sv.replace(' ','_').lower()} for sv in SOURCE_STORAGE_VARIABLES])

        constituents = self.v.model.get_constituents()
        for c in constituents:
            recorders.append([{'recorder':{'RecordingVariable':'Constituents@%s@%s'%(c,cv)},'label':'%s%s'%(c,lbl)} for cv,lbl in zip(CONSTITUENT_VARIABLES,CONSTITUENT_VARIABLE_LABELS)])
        # for cv in CONSTITUENT_VARIABLES:
        #     recorders += [{'RecordingVariable':'Constituents@%s@%s'%(c,cv)} for c in constituents]
        return recorders

    def _configure_key_recorders(self):
        recorders = [{'RecordingElement':re} for re in STANDARD_SOURCE_ELEMENTS] + \
            [{'RecordingVariable':rv} for rv in STANDARD_SOURCE_VARIABLES + SOURCE_STORAGE_VARIABLES]

        constituents = self.v.model.get_constituents()
        for cv in CONSTITUENT_VARIABLES:
            recorders += [{'RecordingVariable':'Constituents@%s@%s'%(c,cv)} for c in constituents]

        self.v.configure_recording(enable=recorders)
        self.progress('Configured recorders')

    def extract_source_results(self,start=None,end=None,batches=False,before_batch=_BEFORE_BATCH_NOP):
        self.current_dest = self.results
        self._ensure()

        recording_batches = self._get_recorder_batches()
        if not batches:
            recording_batches = [[item for sublist in recording_batches for item in sublist]]
        for ix,batch in enumerate(recording_batches):
            before_batch(self,ix,batch)
            self.v.drop_all_runs()

            self.progress('Running batch %d of %d, with %d recorders'%(ix+1,len(recording_batches),len(batch)))
            self.progress('Running to get:\n* '+('\n* '.join([r['label'] for r in batch])))

            recorders = [r['recorder'] for r in batch]        
            self.v.configure_recording(enable=recorders,disable=[{}])

            self.v.model.simulation.configure_assurance_rule(level='Warning',category='Data Sources')

            self.v.run_model(start=start,end=end)
            self.progress('Simulation done.')

            run_summary = self.v.retrieve_run()
            results_df = run_summary['Results'].as_dataframe()
            self.progress('Got %d results'%len(results_df))

            for r in batch:
                retriever = r.get('retriever',r['recorder'])
                name_fn = veneer.name_for_location
                rv = retriever.get('RecordingVariable','')
                if rv.endswith(' Flow') or r['label'].endswith('generation'):
                    name_fn = veneer.name_for_fu_and_sc
                ts_results = self.v.retrieve_multiple_time_series(run_data=run_summary,criteria=retriever,name_fn=name_fn)
                self.write_csv(r['label'],ts_results)

            # self.write_csv('results',results_df)
        return
        # variables = set(results_df.RecordingVariable)

        # downstream = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Downstream Flow Volume'},name_fn=veneer.name_for_location)
        # self.progress('Got downstream flow')
        # self.write_csv('downstream_vol',downstream)

        # upstream = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Upstream Flow Volume'},name_fn=veneer.name_for_location)
        # self.progress('Got upstream flow')
        # self.write_csv('upstream_vol',upstream)

        # runoff = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Quick Flow'},name_fn=veneer.name_for_fu_and_sc)
        # self.write_csv('runoff',runoff)

        # baseflow = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Slow Flow'},name_fn=veneer.name_for_fu_and_sc)
        # self.write_csv('baseflow',baseflow)

        # totalflow = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Total Flow'},name_fn=veneer.name_for_fu_and_sc)
        # self.write_csv('totalflow',totalflow)

        # def download_constituent_outputs(suffix,fn_suffix,name_fn=veneer.name_for_location):
        #     constituent_variables = [v for v in variables if v.startswith('Constituents@') and v.endswith(suffix)]
        #     self.progress(constituent_variables)
        #     for cv in constituent_variables:
        #         con = cv.split('@')[1].replace(' ','')
        #         self.progress(con)
        #         ts = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':cv},name_fn=name_fn)
        #         self.write_csv(con+fn_suffix,ts)
        #         self.progress('Downloaded %s %s'%(con,fn_suffix))

        # for cv,lbl in zip(CONSTITUENT_VARIABLES,CONSTITUENT_VARIABLE_LABELS):
        #     download_constituent_outputs(cv,lbl)

        # for sv in SOURCE_STORAGE_VARIABLES:
        #     ts = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':sv},name_fn=veneer.name_for_location)
        #     self.write_csv(sv.replace(' ','_').lower(),ts)


UNIT_CONVERSIONS={
    'None':DefaultDict(lambda: None),
    'ML':{
        'm3':1e3
    },
    'd':{
        's':86400
    }
}

def conversion_factor(src_units,dest_units,lbl):
    dest_units = dest_units.split('/')
    src_units = src_units.split('/')

    conversions = [1.0 if src==dest else UNIT_CONVERSIONS[src][dest] for src,dest in zip(src_units,dest_units)]
    if None in conversions:
        lbl = '' if lbl is None else f' in {lbl}'
        logger.warn(f'Missing source units {lbl}, assuming no conversion factor')
        return 1.0

    if len(conversions)>1:
        return conversions[0]/conversions[1]
    return conversions[0]

def ensure_units(dataframe,dest_units,lbl=None):
    for col in dataframe.columns:
        factor = conversion_factor(dataframe[col].units,dest_units,lbl)
        if factor != 1.0:
            logger.info(f'Converting {col} with factor {factor}')
            dataframe[col] *= factor
            dataframe[col].units = dest_units
    return dataframe

def _base_arg_parser(model=True):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',help='JSON configuration file')
    parser.add_argument('-e','--extractedfiles',help='Parent directory for model files extracted from Source',default='.')
    if model:
      parser.add_argument('model',type=str,help='Name of model to be converted (eg the name of the RSPROJ file without the file extension)')

    return parser

def _arg_parser():
    parser = _base_arg_parser()
    parser.add_argument('-s','--sourceversion',help='Source version number',default='4.5.0')
    parser.add_argument('--port',help='Port number of running Veneer instance',type=int,default=0)
    parser.add_argument('-p','--plugins',type=str,nargs='*',help='Path to plugin (DLL) file to load',default=[])
    parser.add_argument('-v','--veneerpath',help='Path (directory) containing Veneer command line files. If not provided, or not existing, will attempt to create using sourceversion and build paths')
    parser.add_argument('-b','--buildpath',help='Path (directory) containing Veneer builds')
    return parser

def _parsed_args(parser):
    args = vars(parser.parse_args())
    config_path = args['config'] or ''
    if config_path:
        print(f'Reading from {config_path}')
        with open(config_path,'r') as fp:
            config_vals = json.load(fp)
            for k,v in args.items():
                if (k not in config_vals) or (v != parser.get_default(k)):
                    config_vals[k]=v
            # config_vals.update(args)
            args = config_vals
    else:
        print('No config file')
    return args

# def _get_veneer_command_line(veneer=None,sourceversion=None,buildpath=None,**kwargs):
#     pass

def _get_veneer(model_fn,port,veneerpath,sourceversion,buildpath,plugins,**kwargs):
    if port:
        client = veneer.Veneer(port)
        def _nop():pass
        def _get():
            return client
        return _get, _nop

    if not veneerpath:
        veneerpath = tempfile.mkdtemp('-veneer-extract-config')

    exe_path = os.path.join(veneerpath,VENEER_EXE_FN)
    if not os.path.exists(exe_path):
        cmd = create_command_line(buildpath,source_version=sourceversion,dest=veneerpath)
        assert str(cmd)==exe_path
    
    process_details = {}
    def start():
        proc,port = manage.start(model_fn,1,debug=True,veneer_exe=exe_path,additional_plugins=plugins)
        process_details['pid'] = proc[0]
        client = veneer.Veneer(port[0])
        return client

    def stop():
        logging.info('Stopping veneer process with PID:%d',process_details['pid'])
        kill_all_now([process_details['pid']])

    return start,stop

def extract(converter_constructor,model,extractedfiles,**kwargs): # port,buildpath,veneerpath,sourceversion,plugins
    print(f'Extracting {model}...')
    # BUT... how to also use
    for k,v in kwargs.items():
        print(k,v)

    model_fn = model
    model = model.split('/')[-1].split('\\')[-1].split('.')[0]
    start_veneer, stop_veneer = _get_veneer(model_fn,**kwargs)

    veneer_client = start_veneer() #port,buildpath,veneerpath,sourceversion,plugins)

    scenario_info = veneer_client.scenario_info()
    print(scenario_info)

    converter = converter_constructor(veneer_client,
                                      os.path.join(extractedfiles,model),
                                      progress=logging.info)

    converter.extract_source_config()

    def between_batches(extractor,ix,batch):
        print('Running batch %d for %s'%(ix,model))
        if ix > 0:
            stop_veneer()
            v = start_veneer()
            v.drop_all_runs()
            converter.v = v

    converter.extract_source_results(batches=True,before_batch=between_batches)

    stop_veneer()

    # TODO: Delete veneer command line

if __name__=='__main__':
    args = _parsed_args(_arg_parser())
    extract(SourceExtractor,**args)
