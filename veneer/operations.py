import pandas as pd
import numpy as np

SERIALISE_TS='''
def serialise_ts(ts):
    dates = [ts.timeForItem(i) for i in range(ts.Count)]
    values = ts.toArray()
    return list(zip(dates,values))
'''

GET_OVERRIDES_TEMPLATE='''
items = scenario.OperationsMaster.NetworkElementOperationsDataItems
%s

%s
result = {(item.NetworkElement.Name,item.RefItem.itemName,item.Units):serialise_ts(item.%sMapping.OverrideTimeSeries) for item in items}
'''

FIND_OVERRIDE_TEMPLATE='''
def get_override_time_series(location,variable):
    item = scenario.OperationsMaster.NetworkElementOperationsDataItems.First(lambda i: i.NetworkElement.Name==location and i.RefItem.itemName==variable)
    return item.%sMapping.OverrideTimeSeries

'''


class VeneerOperationsActions(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython
    
    def get_overrides(self,time_period,keep_nulls=False,locations=None,variables=None):
        '''
        Retrieve a dataframe of operator overrides from the current scenario:

        Parameters:
        v: the Veneer client object pointing to an operations scenario
        time_period: the time period to retrieve overrides for, should be 'Historic' or 'Forecast'
        keep_nulls: if True, keep columns where there are no overrides
        locations: a list of locations to retrieve overrides for, or None to retrieve all locations
        variables: a list of variables to retrieve overrides for, or None to retrieve all variables

        Returns:
        A pandas DataFrame with the overrides, indexed by date, and with a MultiIndex of location, variable and units
        '''
        loc_query = ''
        if locations:
            loc_query = 'items = items.Where(lambda i: i.NetworkElement.Name in %s)\n'%str(locations)
        var_query = ''
        if variables:
            var_query = 'items = items.Where(lambda i: i.RefItem.itemName in %s)\n'%str(variables)

        ts = self._ironpy._safe_run(self._ironpy._init_script()+SERIALISE_TS+GET_OVERRIDES_TEMPLATE%(loc_query,var_query,time_period))
        parsed = self._ironpy.simplify_response(ts['Response'])
        series = {k:pd.Series([v[1] for v in ts],index=[v[0] for v in ts]) for k,ts in parsed.items()}
        series = {k:v.replace(-9999,np.nan) for k,v in series.items()}
        if not keep_nulls:
            is_empty = {k:not len(v.dropna()) for k,v in series.items()}
            series = {k:v for k,v in series.items() if not is_empty[k]}
        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df.index,dayfirst=True)
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
        getter_script = FIND_OVERRIDE_TEMPLATE%time_period
        script = self._ironpy._init_script()
        script += getter_script
        for col in overrides.columns:
            location = col[0]
            variable = col[1]
            # ignore units, assume matching
            script += f"ts = get_override_time_series('{location}','{variable}')\n"
            series = overrides[col].dropna()
            for ix, val in series.items():
                script += f"dt = System.DateTime({ix.year},{ix.month},{ix.day})\n"
                script += f"ts[dt] = {val}\n"
            
        return self._ironpy._veneer.run_server_side_script(script)
    
    


