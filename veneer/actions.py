'''
High level actions
'''
from .utils import _safe_filename

def switch_data_source(v,from_set,to_set,remove_original=True):
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

"""%(v.model._init_script(),from_set,to_set)
	if remove_original:
		switch_input_sets_script += "\ndm.RemoveGroup(orig_item)"
	return v.model.run_script(switch_input_sets_script)

def enable_streaming(v,fn,overwrite='Fail'):
	'''
	Configure Source to stream time series results to a HDF5 file (fn)

	Optionally specify behaviour for when the file exists (overwrite string):
	 * 'Fail' - default - Don't run the simulation if the output file already exists,
	 * 'Overwrite' - attempt to overwrite the existing output file. 
	                 Note This option will fail if the existing file is locked, such as if the previous run is
					 still loaded in Source
     * 'Increment' - Change the filename by adding an incrementing integer
	'''
	script='''
%s
import FlowMatters.Source.HDF5IO.StreamingOutputManager as StreamingOutputManager
import FlowMatters.Source.HDF5IO.StreamingOutputOverwriteOption as StreamingOutputOverwriteOption
streamer = StreamingOutputManager.EnableStreaming(scenario,"%s")
streamer.OverwriteOption = StreamingOutputOverwriteOption.%s
'''%(v.model._init_script(),_safe_filename(fn),overwrite)
	return v.model.run_script(script)

def disable_streaming(v):
	script = v.model._init_script() + '''
import FlowMatters.Source.HDF5IO.StreamingOutputManager as StreamingOutputManager
StreamingOutputManager.DisableStreaming(scenario)
'''
	return v.model.run_script(script)