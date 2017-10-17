'''
High level actions
'''
from .utils import _safe_filename

def switch_data_source(v,from_set,to_set):
	switch_input_sets_script="""
%s
dm = scenario.Network.DataManager
orig_item = dm.DataGroups.First(lambda g: g.Name=='%s')
dest_item = dm.DataGroups.First(lambda g: g.Name=='%s')

def transfer_usages(orig_item,dest_item):
  for orig_detail in orig_item.DataDetails:
    dest_detail = dest_item.DataDetails.First(lambda dd:dd.Name==orig_detail.Name)

    for usage in orig_detail.Usages:
      dest_detail.Usages.Add(usage)
    orig_detail.Usages.Clear();

transfer_usages(orig_item,dest_item)
dm.RemoveGroup(orig_item)
"""%(v.model._initScript(),from_set,to_set)
	return v.model.run_script(switch_input_sets_script)

def enable_streaming(v,fn):
	script='''
%s
import FlowMatters.Source.HDF5IO.StreamingOutputManager as StreamingOutputManager
StreamingOutputManager.EnableStreaming(scenario,"%s")
'''%(v.model._initScript(),_safe_filename(fn))
	return v.model.run_script(script)

def disable_streaming(v):
	script = v.model._initScript() + '''
import FlowMatters.Source.HDF5IO.StreamingOutputManager as StreamingOutputManager
StreamingOutputManager.DisableStreaming(scenario)
'''
	return v.model.run_script(script)