import sys
import inspect

from types import MethodType
from .utils import SearchableList

_WU_ICON='/resources/WaterUserNodeModel'

def _node_id(node):
    if hasattr(node,'keys'):
        if 'properties' in node:
            return node['properties'].get('id',node['id'])
        return node['id']
    return node

def _feature_list(lst):
    return SearchableList(lst,nested=['properties'])

def network_downstream_links(self,node_or_link):
    '''
    Find all the links in the network that are immediately downstream of a given node.

    Parameters:

    * node_or_link  - the node to search on. Expects the node feature object
    '''
    features = self['features']
    source = features.find_by_id(_node_id(node_or_link))[0]
    if source['properties']['feature_type']=='node':
        node = _node_id(source)
    else:
        node = source['properties']['to_node']

    links = features.find_by_feature_type('link')
    return links.find_by_from_node(node)

def network_upstream_links(self,node_or_link):
    '''
    Find all the links in the network that are immediately upstream of a given node.

    Parameters:

    * node_or_link  - the node or link to search on. Expects the node feature object
    '''   
    features = self['features']
    source = features.find_by_id(_node_id(node_or_link))[0]
    if source['properties']['feature_type']=='node':
        node = source['id']
    else:
        node = source['properties']['from_node']

    links = features.find_by_feature_type('link')
    return links.find_by_to_node(node)

def network_node_names(self):
    '''
    Return a list of node names as found within the network.

    Example:

    v = Veneer()
    network = v.network()
    node_names = network.node_names()
    '''
    return self['features'].find_by_feature_type('node')._all_values('name')

def network_models(self):
    '''
    Return a list of models within the network.

    Extracts this information through the icon resource attribute.

    Example:

    v = Veneer()
    network = v.network()
    node_models = network.models()
    '''
    nodes = self['features'].find_by_feature_type('node')._unique_values('icon')

    node_elements = []
    for n in nodes:
        node_elements.append(n.split('/')[-1])

    return node_elements

def find_network_model(self, model_name):
    '''
    Find information about a node type by its resource name

    model_name: str, name of node type

    Example:

    v = Veneer()
    network = v.network()
    waterusers = network.find_network_model('WaterUserNodeModel')
    '''
    return self['features'].find_by_icon('/resources/'+model_name)

def network_outlet_nodes(self):
    '''
    Find all the nodes in the network that are outlets - ie have no downstream nodes.

    Note: Filters out Water User Nodes

    Returns each node as a feature (ie with fields 'geometry' and 'properties')
    '''
    features = self['features']
    nodes = features.find_by_feature_type('node')

    no_downstream = _feature_list([n for n in nodes if len(self.downstream_links(n))==0])
    no_downstream_excluding_water_user = no_downstream.find_by_icon(_WU_ICON,op='!=')

    water_users = nodes.find_by_icon(_WU_ICON)
    water_user_links = _feature_list([upstream_link for wu in water_users \
                                                    for upstream_link in self.upstream_links(wu)])

    nodes_to_water_users = [nodes.find_by_id(link['properties']['from_node'])[0] for link in water_user_links]

    nodes_with_only_water_users = [n for n in nodes_to_water_users if len(self.downstream_links(n))==1]

    return _feature_list(no_downstream_excluding_water_user._list+nodes_with_only_water_users)

def network_as_dataframe(self):
    try:
        from geopandas import GeoDataFrame
        result = GeoDataFrame.from_features(self['features'])
        result['id'] = [f['id'] for f in self['features']]
        return result
    except Exception as e:
        print('Could not create GeoDataFrame. Using regular DataFrame.')
        print(str(e))

    return self['features'].as_dataframe()

def network_partition(self,key_features,new_prop):
    '''
    Partition the network by a list of feature names (key_features).

    Add property (new_prop) to each feature identifying next downstream feature from key_features.

    Features in key_features are assigned to their own group.

    Features with no downstream key_feature (eg close to outlets) are attributed with their outlet node
    '''
    features = self['features']

    def attribute_next_down(feature):
        if new_prop in feature['properties']:
            return feature['properties'][new_prop]

        if feature['properties']['name'] in key_features:
            feature['properties'][new_prop] = feature['properties']['name']
            return feature['properties']['name']

        f_type = feature['properties']['feature_type']

        if f_type=='catchment':
            ds_feature_id = feature['properties']['link']
        elif f_type=='link':
            ds_feature_id = feature['properties']['to_node']
        else: # f_type=='node'
            downstream_links = self.downstream_links(feature)
            if len(downstream_links)==0:
                # Outlet and we didn't find one of the key features...
                feature['properties'][new_prop] = feature['properties']['name']
                return feature['properties']['name']
            elif len(downstream_links)>1:
                for l in downstream_links:
                    key = attribute_next_down(l)
                    if key in key_features:
                        # If this link leads to a key_feature, return that name
                        feature['properties'][new_prop] = key
                        return key
                # All downstream links terminate without reaching a key_feature
                # Return the first one
                feature['properties'][new_prop] = downstream_links[0]['properties'][new_prop] 
                return downstream_links[0]['properties'][new_prop]

            # Just one downstream link, usual case
            ds_feature_id = downstream_links[0]['id']

        ds_feature = features.find_by_id(ds_feature_id)[0]
        key = attribute_next_down(ds_feature)
        feature['properties'][new_prop] = key
        return key

    for f in features:
        attribute_next_down(f)

def network_upstream_features(self,node):
    result = []
    links = self.upstream_links(node)
    result += links._list
    for l in links:
        catchment = self['features'].find_by_link(l['id'])
        if catchment is not None:
            result += catchment._list
        
        upstream_node = self['features'].find_by_id(l['properties']['from_node'])[0]
        result.append(upstream_node)
        result += self.upstream_features(upstream_node['id'])._list
    return SearchableList(result,nested=['properties'])

def network_plot(self,nodes=True,links=True,catchments=True,ax=None,zoom=0.05):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import numpy as np
    v = self['_v']

    icons = set(self['features'].find_by_feature_type('node')._select(['icon']))

    icon_images = {i:plt.imread(v.url(i+v.img_ext)) for i in icons}

    if ax is None:
        ax = plt.gca()

    df = self.as_dataframe()

    if catchments:
        params = catchments if hasattr(catchments,'keys') else {}
        df[df.feature_type=='catchment'].plot(ax=ax,**params)
    
    if links:
        params = links if hasattr(links,'keys') else {}
        df[df.feature_type=='link'].plot(ax=ax,**params)

    if not nodes:
        return ax

    df_nodes = df[df.feature_type=='node']
    points = df_nodes.geometry.tolist()
    x = [p.x for p in points]
    y = [p.y for p in points]
    x, y = np.atleast_1d(x, y)
    nodes = []
    for x0, y0,icon in zip(x, y,df_nodes.icon):
        im = OffsetImage(icon_images[icon], zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        nodes.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return ax

def network_path_between(self,from_feature,to_feature):
    from_id = _node_id(from_feature)
    to_id = _node_id(to_feature)

    if from_id==to_id:
        return SearchableList([])

    feature = self['features'].find_one_by_id(from_id)
    if from_id.startswith('/network/catchments'):
        immediate_down = self['features'].find_one_by_id(feature['properties']['link'])
    elif from_id.startswith('/network/link'):
        immediate_down = self['features'].find_one_by_id(feature['properties']['to_node'])
    else: # node
        possible_downs = self.downstream_links(feature)
        for pd in possible_downs:
            path_via = self.path_between(pd,to_feature)
            if path_via is not None:
                return SearchableList([pd]+path_via._list,nested=['properties'])
        return None
    path_to = self.path_between(immediate_down,to_feature)
    if path_to is None:
        return None
    return SearchableList([immediate_down]+path_to._list,nested=['properties'])

def add_network_methods(target):
    '''
    Attach extension methods to an object that represents a Veneer network.
    Note: The 'network_' prefix will be removed from method names.

    target: Veneer network object to attach extension methods to.
    '''
    import veneer.extensions as extensions # Import self to inspect available functions

    # Generate dict of {function name: function}, skipping this function
    this_func_name = sys._getframe().f_code.co_name
    funcs = inspect.getmembers(extensions, inspect.isfunction)
    funcs = dict((func_name, func) for func_name, func in funcs
                  if func_name != this_func_name
            )

    # Assign functions to target, removing the 'network_' prefix
    for f_name, f in funcs.items():
        if f_name.startswith('network_'):
            f_name = f_name.replace('network_', '')
        setattr(target, f_name, MethodType(f, target))

def _apply_time_series_helpers(dataframe):
    from pandas import DataFrame
#        import types

    def by_wateryear(df_self,start_month,start_day=1):
        '''
        Group timesteps by water year
        '''
        def water_year(d):
            if (d.month==start_month and d.day >= start_day) or (d.month>start_month):
                return "%d-%d"%(d.year,d.year+1)
            return "%d-%d"%(d.year-1,d.year)
        water_year = df_self.index.map(water_year)

        result = df_self.groupby(water_year)
        return result

    def of_month(df_self,month):
        '''
        Filter by timesteps in month (integer)
        '''
        result = df_self[df_self.index.month==month]
        self._apply_time_series_helpers(result)
        return result

    def by_month(df_self):
        '''
        Group timesteps by month
        '''
        result = df_self.groupby(df_self.index.month)
        return result

    def of_year(df_self,year):
        '''
        Filter by timesteps in year (integer)
        '''
        result = df_self[df_self.index.year==year]
        self._apply_time_series_helpers(result)
        return result

    def by_year(df_self):
        '''
        Group timesteps by year
        '''
        result = df_self.groupby(df_self.index.year)
        return result

    meths = {
        'by_wateryear':by_wateryear,
        'of_month':of_month,
        'by_month':by_month,
        'of_year':of_year,
        'by_year':by_year,
    }

    dataframe.__class__ = type('VeneerDataFrame',(DataFrame,),meths)
