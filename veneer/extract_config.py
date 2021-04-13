import io
import json
import os
import pandas as pd
from veneer.actions import get_big_data_source
import veneer
from string import Template

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

class SourceExtractor(object):
    def __init__(self,v,dest,results=None,climate_data_sources=['Climate Data'],progress=print):
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

    def _extract_runoff_configuration(self):
        self._extract_models_and_parameters('catchment.runoff','runoff_models','rr')
        # runoff_models = self.v.model.catchment.runoff.model_table()
        # runoff_params = self.v.model.catchment.runoff.tabulate_parameters()
        # for model_type, table in runoff_params.items():
        #     self.write_csv('rr-%s'%model_type,table)
        # self.write_csv('runoff_models',runoff_models)

        # runoff_inputs = self.v.model.catchment.runoff.tabulate_inputs('Dynamic_SedNet.Models.Rainfall.DynSedNet_RRModelShell')

        self.progress('!!!! Skipping climate data')
        # self.progress('Getting climate data')
        # climate = pd.DataFrame()
        # for ds_name in self.climate_data_sources:
        #     ds = get_big_data_source(self.v,ds_name,self.data_sources,self.progress)
        #     climate = pd.concat([climate,ds],axis=1)

        # self.write_csv('climate',climate)

    def _extract_models_and_parameters(self,path,model_fn,param_prefix):
        source = self.v.model
        for p in path.split('.'):
            source = getattr(source,p)
        models = source.model_table()
        params = source.tabulate_parameters()
        self.write_csv(model_fn,models)
        for model_type, table in params.items():
            self.write_csv('%s-%s'%(param_prefix,model_type),table)

    def _extract_routing_configuration(self):
        self._extract_models_and_parameters('link.routing','routing_models','fr')
        self._extract_models_and_parameters('link.constituents','transportmodels','cr')

        # link_models = self.v.model.link.routing.model_table()
        # link_params = self.v.model.link.routing.tabulate_parameters() #'RiverSystem.Flow.StorageRouting')

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
            d = self.v.model.node.water_users.get_param_values('DemandModel.Name',nodes=wu)[0]
        #     assert len(d) == 1
        #     d = d[0]
            if d.startswith('Time Series Demand #'):
                demands[wu] = self.v.model.node.water_users.get_data_sources('DemandModel.Order',nodes=wu)[0]
            elif d.startswith('Monthly Pattern #'):
                txt = self.v.model.node.water_users.get_param_values('DemandModel.Quantities',nodes=wu)[0]       
                demands[wu] = pd.DataFrame([{'month':ln[0],'volume':float(ln[1])} for ln in [ln.split(' ') for ln in txt.splitlines()]])
            else:
                raise Exception('Unsupported demand model: %s'%d)

        for node,demand in demands.items():
            # print(demand)
            # demand = demand.replace('%2F','/')
            # print(node,"'%s'"%demand)
            if isinstance(demand,pd.DataFrame):
                self.write_csv('monthly-pattern-demand-%s'%node,demand)
                continue

            if demand == '':
                print('No demand time series configured for node: %s'%node)
                continue

            data_source = demand.split('/')[2]
            # print(data_source)
            ds = self.v.data_source(data_source)['Items'][0]['Details']
            self.write_csv('timeseries-demand-%s'%node,ds)

    def _extract_storage_configuration(self):
        params = self.v.model.node.storages.tabulate_parameters()

        if not len(params):
            self.progress('No storages in model')
            return

        self.progress('Extracting information for %d storages'%len(params))
        params = params['RiverSystem.Nodes.StorageNodeModel']

        self.write_csv('storage_params',params)

        for storage in list(params['NetworkElement']):
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
                    print('No units on ',col)
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
        played_constituents = played_constituents['RiverSystem.Constituents.ConstituentPlayedValue']

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
                ts_results = self.v.retrieve_multiple_time_series(run_data=run_summary,criteria=retriever,name_fn=veneer.name_for_location)
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


