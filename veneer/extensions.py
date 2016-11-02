
from types import MethodType
from .utils import SearchableList

def network_downstream_links(self,node):
    '''
    Find all the links in the network that are immediately downstream of a given node.

    Parameters:

    * node  - the node to search on. Expects the node feature object
    '''
    features = self['features']
    links = features.find_by_feature_type('link')
    return links.find_by_from_node(node['id'])

def network_upstream_links(self,node):
    '''
    Find all the links in the network that are immediately downstream of a given node.

    Parameters:

    * node  - the node to search on. Expects the node feature object
    '''
    features = self['features']
    links = features.find_by_feature_type('link')
    return links.find_by_to_node(node['id'])

def network_find_outlets(self):
    '''
    Find all the nodes in the network that are outlets - ie have no downstream nodes.

    Note: Filters out Water User Nodes

    Returns each node as a feature (ie with fields 'geometry' and 'properties')
    '''
    features = self['features']
    nodes = features.find_by_feature_type('node')
    no_downstream = SearchableList([n for n in nodes if len(self.downstream_links(n))==0],nested=['properties'])
    return no_downstream.find_by_icon('/resources/WaterUserNodeModel',op='!=')

def network_as_dataframe(self):
    from geopandas import GeoDataFrame
    result = GeoDataFrame.from_features(self['features'])
    result['id'] = [f['id'] for f in self['features']]
    return result

def add_network_methods(target):
    target.downstream_links = MethodType(network_downstream_links,target)
    target.upstream_links = MethodType(network_upstream_links,target)
    target.outlet_nodes = MethodType(network_find_outlets,target)
    target.as_dataframe= MethodType(network_as_dataframe,target)
