import sys
import inspect

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
    Find all the links in the network that are immediately upstream of a given node.

    Parameters:

    * node  - the node to search on. Expects the node feature object
    '''
    features = self['features']
    links = features.find_by_feature_type('link')
    return links.find_by_to_node(node['id'])

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
    no_downstream = SearchableList([n for n in nodes if len(self.downstream_links(n))==0],nested=['properties'])
    return no_downstream.find_by_icon('/resources/WaterUserNodeModel',op='!=')

def network_as_dataframe(self):
    from geopandas import GeoDataFrame
    result = GeoDataFrame.from_features(self['features'])
    result['id'] = [f['id'] for f in self['features']]
    return result

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
