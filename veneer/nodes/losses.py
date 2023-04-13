import pandas as pd
import numpy as np
from veneer.server_side import VeneerNetworkElementActions

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
        return self.tabulate_list_values('lossFct',['Inflow','Loss'],['Key','Value'],nodes=[node])
