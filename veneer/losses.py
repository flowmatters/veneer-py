from types import MethodType
import pandas as pd
import numpy as np
from .server_side import VeneerNetworkElementActions
from .utils import _quote_string

GET_LOSS_TABLE_SCRIPTLET='''
ignoreExceptions=False
fn = target.lossFct
for row in fn:
    result.append((row.Key,row.Value))
'''

class VeneerLossNodeActions(VeneerNetworkElementActions):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        self._name_accessor = self.node_actions._name_accessor
        super(VeneerLossNodeActions, self).__init__(node_actions._ironpy)
    def _build_accessor(self, parameter=None, nodes=None):
        return self.node_actions._build_accessor(parameter,nodes=nodes,node_types='LossNodeModel')

    def loss_table(self,node):
        '''
        Retrieve the Loss table for a given loss node
        '''
        code = GET_LOSS_TABLE_SCRIPTLET
        vals = self.apply(code,init='[]',nodes=[node])
        return pd.DataFrame(vals,columns=['inflow','loss'])
