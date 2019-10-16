


def name_time_series(result):
    '''
    Name the retrieved time series based on the full name of the time series (including variable and location)
    '''
    return result.get('TimeSeriesName', result.get('Name', '%s/%s/%s' % (result['NetworkElement'], result['RecordingElement'], result['RecordingVariable'])))


def name_element_variable(result):
    element = result['NetworkElement']
    variable = result['RecordingVariable'].split(' - ')[-1]
    if variable == 'Flow':
        variable = result['RecordingElement'].split(' - ')[-1]
    return '%s:%s' % (element, variable)


def name_for_variable(result):
    '''
    Name the retrieved time series based on the variable only.

    Useful when retrieving multiple time series from one network location.
    '''
    return result['RecordingVariable']


def name_for_end_variable(result):
    '''
    Name the retrieved time series based on the final part of the RecordingVariable (eg after the last @)
    '''
    return name_for_variable(result).split('@')[-1]


def name_for_location(result):
    '''
    Name the retrieved time series based on the network location only.

    Useful when retrieving the same variable from multiple locations.
    '''
    return result['NetworkElement']


def name_for_fu_and_sc(result):
    '''
    For a FU-based time series, name based on the FU and the subcatchment.

    Note, when retrieving FU based results, you should limit the query to a single FU (or 'Total').
    Due to a quirk in the Source recording system, you'll get the results for all FUs anyway.
    If you don't specify a single FU, the system will make separate requests for each FU and get multiple
    results from each requests (essentially transferring n^2 data).
    '''
    char = ':'
    if not 'TimeSeriesName' in result:
        char = '/'
    return ':'.join(name_time_series(result).split(':')[:2])


def name_for_fu(result):
    '''
    For a FU-based time series, name based on the FU.

    Note, when retrieving FU based results, you should limit the query to a single FU (or 'Total').
    Due to a quirk in the Source recording system, you'll get the results for all FUs anyway.
    If you don't specify a single FU, the system will make separate requests for each FU and get multiple
    results from each requests (essentially transferring n^2 data).
    '''
    return name_for_fu_and_sc(result).split(':')[0]

