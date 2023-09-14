import pandas as pd
from veneer.server_side import VeneerNetworkElementActions

LOAD_PARTITIONING_TABLE_SCRIPTLET = '''
from RiverSystem.Flow import SplitterRelationship
ignoreExceptions=False
part_table = target.EffluentRelationship
part_table.Clear()
%s
result += 1
'''

SET_EFFLUENT_LINK_SCRIPTLET='''
ignoreExceptions=False
target.EffluentLink = scenario.Network.Links.First(lambda l:l.Name=="%s")
'''

class VeneerFlowPartitioningActions(VeneerNetworkElementActions):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        self._name_accessor = self.node_actions._name_accessor
        super(VeneerFlowPartitioningActions, self).__init__(node_actions._ironpy)

    def _build_accessor(self, parameter=None, nodes=None):
        return self.node_actions._build_accessor(parameter,nodes=nodes,
                                                splitter=True,node_types='RegulatedEffluentPartitioner')

    def effluent_relationship(self,node):
        '''
        Retrieve the effluent relationship table for a given node
        '''
        return self.tabulate_list_values('EffluentRelationship',['UpstreamFlow','MinimumEffluent','MaximumEffluent'],nodes=[node])

    def load_partitioning_table(self,part_table,nodes):
        part_text = '\n'.join(['part_table.Add(SplitterRelationship(UpstreamFlow=%f,MinimumEffluent=%f,MaximumEffluent=%f))'%(r['UpstreamFlow'],r['MinimumEffluent'],r['MaximumEffluent']) for _,r  in part_table.iterrows()])
        code = LOAD_PARTITIONING_TABLE_SCRIPTLET%part_text
        return self.apply(code,init='0',nodes=nodes)

    def set_effluent_link(self,link_name,node):
        code = SET_EFFLUENT_LINK_SCRIPTLET%link_name
        return self.apply(code,nodes=node)

