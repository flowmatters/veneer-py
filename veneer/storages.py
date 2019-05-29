

LOAD_LVA_SCRIPTLET = '''
ignoreExceptions=False
geo = target.StoreGeometry.DefiniedGeometrys[0].Geometry
geo.ClearPoints()
%s
result += 1
'''

class VeneerStorageActions(object):
    def __init__(self,node_actions):
        self.node_actions = node_actions

    def load_lva(self,lva_table,nodes=None):
        '''
        Load a Level/Volume/Area table into one or more storage nodes.

        lva_table: A DataFrame with columns level (in metres), volume (in m^3), and area (in m^2)

        nodes: Option node name or list of node names to apply LVA too. Default: apply to all storage nodes
        '''
        lva_text = '\n'.join(['geo.Add(%f,%f,%f)'%(r['level'],r['area'],r['volume']) for _,r  in lva_table.iterrows()])
        code = LOAD_LVA_SCRIPTLET%lva_text
        return self.node_actions.apply(code,init='0',node_types='StorageNodeModel',nodes=nodes)

