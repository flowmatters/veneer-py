import pandas as pd
import numpy as np

SERIALISE_TS='''
def serialise_ts(ts):
    dates = [ts.timeForItem(i) for i in range(ts.Count)]
    values = ts.ToArray()
    return list(zip(dates,values))
'''

SECONDARY_NAME='''
def secondary_override_name(item):
    primary_name = item.NetworkElement.Name
    secondary = item.RefItem.PermanentTarget
    secondary_name = '-'
    if hasattr(secondary,'Name'):
        secondary_name = secondary.Name
    elif hasattr(secondary,'name'):
        secondary_name = secondary.name
    elif hasattr(secondary,'Link'):
        secondary_name = secondary.Link.Name
    # else:
    #     secondary_name = secondary.GetType().Name + ':' + str(secondary)
    if secondary_name == primary_name:
        return '-'
    return secondary_name
'''

GET_OVERRIDES_TEMPLATE='''
items = scenario.OperationsMaster.NetworkElementOperationsDataItems
%s

%s

%s
result = {(item.NetworkElement.Name,item.RefItem.itemName,secondary_override_name(item),item.Units):serialise_ts(item.%sMapping.OverrideTimeSeries) for item in items}
'''

FIND_OVERRIDE_TEMPLATE='''
def get_override_time_series(location,variable,secondary):
    if secondary is None:
        search = lambda i: i.NetworkElement.Name==location and i.RefItem.itemName==variable
    else:
        search = lambda i: i.NetworkElement.Name==location and i.RefItem.itemName==variable and secondary_override_name(i)==secondary
    item = scenario.OperationsMaster.NetworkElementOperationsDataItems.First(search)
    return item.%sMapping.OverrideTimeSeries
'''

CLEAR_OVERRIDES_TEMPLATE='''
items = scenario.OperationsMaster.NetworkElementOperationsDataItems
cleared = 0
touched = 0
%s
for item in items:
    ts = item.%sMapping.OverrideTimeSeries
    vals = ts.ToArray()
    n = 0
    for i in range(ts.Count):
        if vals[i] != -9999:
            t = ts.timeForItem(i)
            if (start is None or t >= start) and (end is None or t <= end):
                ts[t] = -9999
                n = n + 1
    if n > 0:
        touched = touched + 1
        cleared = cleared + n
result = {'items':touched,'cells':cleared}
'''



TRAVEL_TIME_OFFSETS_TEMPLATE='''
import clr
import System
import RiverSystem.Operations.TabularColumnFormatting as TCF
from System.Reflection import BindingFlags

def _tt_name(e):
    return e.Name if hasattr(e,'Name') else str(e)

seq = TCF.NodeLinkNetworkOptimisedSequencer()
net = scenario.Network
offsets = seq.GetDisplayColumnTravelTimes(net, %s)
order = seq.NetworkElementOrder(net)

_MIN = System.Int32.MinValue
element_offsets = {}
for kv in offsets:
    val = kv.Value
    element_offsets[_tt_name(kv.Key)] = None if val == _MIN else val

fld = seq.GetType().GetField('_linkTravelTimes', BindingFlags.NonPublic | BindingFlags.Instance)
link_tt = fld.GetValue(seq)
link_travel_times = {}
for kv in link_tt:
    link_travel_times[_tt_name(kv.Key)] = kv.Value

result = {
    'element_offsets': element_offsets,
    'link_travel_times': link_travel_times,
    'order': [_tt_name(e) for e in order],
    'use_max': %s,
}
'''


class VeneerOperationsActions(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython

    def get_overrides(self,time_period,keep_nulls=False,locations=None,variables=None,secondary=None):
        '''
        Retrieve a dataframe of operator overrides from the current scenario:

        Parameters:
        v: the Veneer client object pointing to an operations scenario
        time_period: the time period to retrieve overrides for, should be 'Historic' or 'Forecast'
        keep_nulls: if True, keep columns where there are no overrides
        locations: a list of locations to retrieve overrides for, or None to retrieve all locations
        variables: a list of variables to retrieve overrides for, or None to retrieve all variables
        secondary: a list of secondary locations to filter by, or None to retrieve all.
                   Typically used to filter by particular release paths

        Returns:
        A pandas DataFrame with the overrides, indexed by date, and with a MultiIndex of location, variable, secondary location and units
        '''
        loc_query = ''
        if locations:
            loc_query = 'items = items.Where(lambda i: i.NetworkElement.Name in %s)\n'%str(locations)
        var_query = ''
        if variables:
            var_query = 'items = items.Where(lambda i: i.RefItem.itemName in %s)\n'%str(variables)

        secondary_location_query = ''
        if secondary:
            secondary_location_query = 'items = items.Where(lambda i: secondary_override_name(i) in %s)\n'%str(secondary)

        ts = self._ironpy._safe_run(self._ironpy._init_script()+
                            SERIALISE_TS+
                            SECONDARY_NAME+
                            GET_OVERRIDES_TEMPLATE%(loc_query,var_query,secondary_location_query,time_period))
        parsed = self._ironpy.simplify_response(ts['Response'])
        series = {k:pd.Series([v[1] for v in ts],index=[v[0] for v in ts]) for k,ts in parsed.items()}
        series = {k:v.replace(-9999,np.nan) for k,v in series.items()}
        if not keep_nulls:
            is_empty = {k:not len(v.dropna()) for k,v in series.items()}
            series = {k:v for k,v in series.items() if not is_empty[k]}
        df = pd.DataFrame(series)
        df.index = pd.to_datetime([ts.split(' ')[0] for ts in df.index],dayfirst=True)
        return df.sort_index()

    def apply_overrides(self,overrides,time_period):
        '''
        Apply a set of overrides to the current scenario:

        Parameters:
        v: the Veneer client object pointing to an operations scenario
        overrides: a pandas DataFrame of overrides, indexed by date, with a MultiIndex of location, variable and optionally units
        time_period: the time period to apply overrides to, should be 'Historic' or 'Forecast'

        Note:
        Any 'null' values (eg NaN) in the dataframe are ignored and not passed back to Source.
        To explicitly remove an override for a location/variable/date, set the value to -9999 in the dataframe.

        Returns:
        The result of the server-side script execution
        '''
        getter_script = SECONDARY_NAME+FIND_OVERRIDE_TEMPLATE%time_period
        script = self._ironpy._init_script()
        script += getter_script
        for col in overrides.columns:
            location = col[0]
            variable = col[1]
            secondary = col[2]
            if secondary=='-':
                secondary = None
            else:
                secondary = f"'{secondary}'"
            # ignore units, assume matching
            script += f"ts = get_override_time_series('{location}','{variable}',{secondary})\n"
            series = overrides[col].dropna()
            for ix, val in series.items():
                # Check for NaT or similar, which can't be serialised to .NET, and raise an error if found
                if pd.isnull(ix):
                    raise ValueError(f"Invalid date index {ix} for override at location {location}, variable {variable}, secondary {secondary}. Please ensure all dates are valid and not null.")

                # Check that the value is numeric, and raise an error if not
                if not isinstance(val, (int, float, np.number)):
                    raise ValueError(f"Invalid value {val} for override at location {location}, variable {variable}, secondary {secondary}, date {ix}. Please ensure all override values are numeric.")

                script += f"dt = System.DateTime({ix.year},{ix.month},{ix.day})\n"
                script += f"ts[dt] = {val}\n"

        return self._ironpy._veneer.run_server_side_script(script)

    def clear_overrides(self,time_period,start=None,end=None):
        '''
        Clear every operator override in the given time period.

        Removes all override values (sets them to Source's -9999 'no override'
        sentinel) across every override series in the scenario, in a single
        server-side pass. This is the deterministic counterpart to
        apply_overrides: clear, then apply exactly the values you want.

        Parameters:
        time_period: 'Historic' or 'Forecast'
        start: optional datetime/date — only clear values on or after this date
        end: optional datetime/date — only clear values on or before this date

        Returns:
        A dict: {'items': number of override series touched,
                 'cells': number of values cleared}
        '''
        def _dt(d):
            if d is None:
                return 'None'
            return 'System.DateTime(%d,%d,%d)'%(d.year,d.month,d.day)
        preamble = 'start = %s\nend = %s\n'%(_dt(start),_dt(end))
        script = self._ironpy._init_script()
        script += CLEAR_OVERRIDES_TEMPLATE%(preamble,time_period)
        result = self._ironpy._safe_run(script)
        return self._ironpy.simplify_response(result['Response'])

    def travel_time_offsets(self,use_max=True):
        '''
        Retrieve the network travel-time data used to draw the 'travel time cutout'
        in the Source tabular editor.

        This drives Source's own NodeLinkNetworkOptimisedSequencer (the class the
        tabular editor uses), so the values match what the GUI would paint. Per-link
        travel times are derived from each link's flow-routing model and accumulated
        along the network to give a cumulative travel time (in whole timesteps) for
        every display column (network element).

        Parameters:
        use_max: if True (default), use the maximum travel-time bound of each link's
                 (flow-dependent) routing range; if False, use the minimum bound.
                 Mirrors the sequencer's own useMax option.

        Returns:
        A dict with:
        * 'element_offsets': {element name: cumulative travel time in timesteps (int),
                              or None for elements with no defined travel time (e.g.
                              disconnected / off-path)}
        * 'link_travel_times': {link name: raw per-link travel time (float, in model
                                timesteps)}
        * 'order': list of element names in the tabular editor's upstream->downstream
                   column order
        * 'use_max': the bound used for this query

        To draw the cutout for a selected column, the relative row offset of any other
        column is the difference of their cumulative offsets:
            offset(selected -> col) = element_offsets[col] - element_offsets[selected]

        Note: elements are keyed by name to match the rest of this module; if two
        elements share a name their offsets will collide in the returned dict.
        '''
        use_max_str = 'True' if use_max else 'False'
        script = self._ironpy._init_script() + \
                 TRAVEL_TIME_OFFSETS_TEMPLATE%(use_max_str,use_max_str)
        response = self._ironpy._safe_run(script)
        return self._ironpy.simplify_response(response['Response'])



