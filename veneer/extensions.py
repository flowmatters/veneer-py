import sys
import inspect

from types import MethodType
from .utils import SearchableList, objdict

_WU_ICON='/resources/WaterUserNodeModel'
_EP_ICON='/resources/ExtractionNodeModel'
_RP_ICON='/resources/RegulatedEffluentPartitioner'

def _feature_id(f):
    if hasattr(f,'keys'):
        if 'properties' in f:
            return f['properties'].get('id',f.get('id',None))
        return f.get('id',None)
    return f

def _node_id(node):
    if hasattr(node,'keys'):
        if 'properties' in node:
            return node['properties'].get('id',node.get('id',None))
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
        node = _feature_id(source)
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

def network_headwater_nodes(self):
    return [f for f in self['features'] if f['properties']['feature_type']=='node' and len(self.upstream_links(f))==0]

def network_headwater_links(self):
    hw_nodes = self.headwater_nodes()
    return sum([self.downstream_links(n)._list for n in hw_nodes],[])

def network_headwater_catchments(self):
    hw_links = [l.get('id',l['properties'].get('id',None)) for l in self.headwater_links()]
    hw_catchments = [(l,self['features'].find_by_link(l)) for l in hw_links]
    for lnk,catchments in hw_catchments:
        if not len(catchments):
            print('No catchment draining to link',lnk)
    return [catchments[0] for _,catchments in hw_catchments if len(catchments)]

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

def network_use_schematic_coordinates(self):
    for f in self['features']:
        if f['properties']['feature_type']=='node':
            f['geometry']['coordinates']=f['properties']['schematic_location']
        elif f['properties']['feature_type']=='link':
            from_node = self['features'].find_by_id(f['properties']['from_node'])[0]
            to_node = self['features'].find_by_id(f['properties']['to_node'])[0]
            f['geometry']['coordinates'] = [
                from_node['properties']['schematic_location'],
                to_node['properties']['schematic_location']
            ]

def network_as_dataframe(self):
    try:
        from geopandas import GeoDataFrame
        result = GeoDataFrame.from_features(self['features'])
        if 'id' in result.columns:
            result['veneer_id'] = result['id']
        result['id'] = [_feature_id(f) for f in self['features']]
        return result
    except Exception as e:
        print('Could not create GeoDataFrame. Using regular DataFrame.')
        print(str(e))

    return self['features'].as_dataframe()

def network_partition(self,
                      key_features,
                      new_prop,
                      reverse_water_users=False,
                      reverse_regulated_partitioners=False):
    '''
    Partition the network by a list of feature names (key_features).

    Add property (new_prop) to each feature identifying next downstream feature from key_features.

    Features in key_features are assigned to their own group.

    Features with no downstream key_feature (eg close to outlets) are attributed with their outlet node

    By default, water users are treated as being downstream of the relevation extraction point.
    Set reverse_water_users=True to reverse this behaviour.
    '''
    features = self['features']
    links_to_redo = []

    def attribute_next_down(feature,force_key=None):
        if force_key is None:
            if new_prop in feature['properties']:
                return feature['properties'][new_prop]

            if feature['properties']['name'] in key_features:
                feature['properties'][new_prop] = feature['properties']['name']
                return feature['properties'][new_prop]

        f_type = feature['properties']['feature_type']

        if f_type=='catchment':
            ds_feature_id = feature['properties']['link']
        elif f_type=='link':
            ds_feature_id = feature['properties']['to_node']
        else: # f_type=='node'
            downstream_links = self.downstream_links(feature)
            if len(downstream_links)==0:
                # Outlet and we didn't find one of the key features...
                feature['properties'][new_prop] = force_key or feature['properties']['name']
                return feature['properties'][new_prop]
            elif len(downstream_links)>1:
                is_extraction_point = feature['properties']['icon']==_EP_ICON
                is_regulated_partitioner = feature['properties']['icon']==_RP_ICON
                found_key = None
                links_without_key = []
                for l in downstream_links:
                    key = attribute_next_down(l,force_key=force_key)
                    if key in key_features:
                        found_key = found_key or key
                        feature['properties'][new_prop] = found_key
                    elif is_extraction_point and reverse_water_users:
                        links_without_key.append(l)
                    elif is_regulated_partitioner and reverse_regulated_partitioners:
                        links_without_key.append(l)

                if found_key is None:
                    # All downstream links terminate without reaching a key_feature
                    # Return the first one
                    feature['properties'][new_prop] = downstream_links[0]['properties'][new_prop] 
                    return downstream_links[0]['properties'][new_prop]
                for l in links_without_key:
                    links_to_redo.append((l,found_key))
                return found_key


            # Just one downstream link, usual case
            ds_feature_id = _feature_id(downstream_links[0])

        ds_feature = features.find_by_id(ds_feature_id)[0]
        key = attribute_next_down(ds_feature,force_key=force_key)
        feature['properties'][new_prop] = key
        return key

    for f in features:
        attribute_next_down(f)
    for (l,tag) in links_to_redo:
        attribute_next_down(l,tag)

def network_upstream_features(self,node):
    result = []
    links = self.upstream_links(node)
    result += links._list
    for l in links:
        catchment = self['features'].find_by_link(_feature_id(l))
        if catchment is not None:
            result += catchment._list
        
        upstream_node = self['features'].find_by_id(l['properties']['from_node'])[0]
        result.append(upstream_node)
        result += self.upstream_features(_feature_id(upstream_node))._list
    return add_geodataframe_method_to_list(SearchableList(result,nested=['properties']))

def network_subset_upstream_of(self,node):
    features = self.upstream_features(node)
    node = self['features'].find_by_id(_node_id(node))[0]
    features._list.append(node)
    result = {
        'type':'FeatureCollection',
        '_v':self['_v'],
        'features':features
    }
    result = _extend_network(result)
    return result

def network_subset(self,filter):
    features = self['features'].where(filter)
    result = {
        'type':'FeatureCollection',
        '_v':self['_v'],
        'features':features
    }
    result = _extend_network(result)
    return result

def network_is_downstream_of(self,feature,possibly_upstream_feature):
    path = self.path_between(possibly_upstream_feature,feature)
    return path is not None

def network_catchment_for_link(self,link):
    link = _feature_id(link)
    for f in self['features']:
        if f['properties']['feature_type'] != 'catchment':
            continue
        if f['properties']['link'] == link:
            return f
    return None

def network_plot(self,nodes=True,links=True,catchments=True,ax=None,zoom=0.05,label_nodes=False):
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
    
    if links == 'arrow':
        for _,row in df[df.feature_type=='link'].iterrows():
            start= row.geometry.coords[0]
            end = row.geometry.coords[1]
            delta = (end[0]-start[0],end[1]-start[1])
            plt.arrow(*start,*delta, length_includes_head=True,head_width=abs(max(start))*0.005, head_length=abs(max(start))*0.01)
    else:
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
    for x0, y0,icon,name in zip(x, y,df_nodes.icon,df_nodes.name):
        im = OffsetImage(icon_images[icon], zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        nodes.append(ax.add_artist(ab))
        if label_nodes:
            ax.annotate(name,xy=(x0,y0),xycoords='data',xytext=(x0+0.0001,y0+0.0001),textcoords='data',fontsize=8)
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

def network_connectivity_table(self):
    import pandas as pd
    nw_df = self.as_dataframe()

    nodes = nw_df[nw_df.feature_type=='node']
    relevant_nodes = nodes[~nodes.icon.str.contains('WaterUserNodeModel')]
    links = nw_df[nw_df.feature_type=='link']

    node_names = list(relevant_nodes.name)
    link_names = list(links.name)
    all_names = node_names + link_names
    result = pd.DataFrame(index=all_names,columns=all_names,dtype='i8').fillna(0)
    for _,link in links.iterrows():
        link_name = link['name']
        to_node_id = link.to_node
        to_node = nodes[nodes.id==to_node_id].name.values[0]
        to_node_type = nodes[nodes.id==to_node_id].icon.values[0].split('/')[-1]
        if 'WaterUser' in to_node_type:
            result = result[result.index!=link_name]
            result = result[[c for c in result.columns if c != link_name]]
            continue
        result.loc[link_name,to_node]=1

        from_node_id = link.from_node
        from_node = nodes[nodes.id==from_node_id].name.values[0]
        result.loc[from_node,link_name]=1

    return result

def _extend_network(nw):
    nw = objdict(nw)

    nw['features'] = SearchableList(
        nw['features'], ['geometry', 'properties'])
    add_network_methods(nw)
    return nw

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
        else:
            continue
        setattr(target, f_name, MethodType(f, target))

def add_geodataframe_method_to_list(target):
    def as_dataframe(self):
        feature_collection = {
            'type':'FeatureCollection',
            'features':self._list
        }
        return network_as_dataframe(feature_collection)
    setattr(target, 'as_dataframe', MethodType(as_dataframe, target))
    return target

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
