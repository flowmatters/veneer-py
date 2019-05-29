from types import MethodType
import pandas as pd
import numpy as np
from .server_side import VeneerNetworkElementActions

LOAD_LVA_SCRIPTLET = '''
ignoreExceptions=False
geo = target.StoreGeometry.DefiniedGeometrys[0].Geometry
geo.ClearPoints()
%s
result += 1
'''

RELEASE_NAMESPACE='RiverSystem.Storages.Releases'

ADD_RELEASE_SCRIPTLET='''
ignoreExceptions=False
from %s import %%s
outlet_path = target.OutletPaths%%s
if outlet_path is None:
    continue
new_release = %%s()
new_release.ReleaseItemName = %%s
new_release.OutletPath = outlet_path
prc = target.ProductReleaseContainer
existing_release = prc.Releases.FirstOrDefault(lambda r:r.ReleaseItemName==new_release.ReleaseItemName)
if existing_release is not None:
    prc.RemoveRelease(existing_release)

prc.AddRelease(new_release)
result += 1
'''%RELEASE_NAMESPACE

RELEASE_CLASSES = {
    'ungated_spillway':'ReleaseDamTop',
    'valve':'ReleaseValve',
    'gated_spillway':'ReleaseSpillway',
    'hydropower_valve':'ReleaseHydropowerValve',
    'culvert':'ReleaseCulvert',
    'pump':'ReleasePump'
}

GET_LVA_SCRIPTLET='''
ignoreExcpeptions=False
geo = target.StoreGeometry.DefiniedGeometrys[0].Geometry
for row in geo:
    result.append(row)
'''

GET_RELEASE_TABLE_SCRIPTLET='''
ignoreExceptions=False
prc = target.ProductReleaseContainer
existing_release = prc.Releases.FirstOrDefault(lambda r:r.ReleaseItemName=='%s')
mins = existing_release.MinimumRelease.ToArray()
maxs = existing_release.MaximumRelease.ToArray()
for minimum,maximum in zip(mins,maxs):
    result.append((minimum.Key,minimum.Value,maximum.Key,maximum.Value))
'''

def path_query(path):
    if path is None:
        return ''
    if isinstance(path,int):
        return '[%d]'%path

    return '.FirstOrDefault(lambda op:op.Link.Name=="%s")'%path

class VeneerStorageActions(VeneerNetworkElementActions):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        self._name_accessor = self.node_actions._name_accessor
        super(VeneerStorageActions, self).__init__(node_actions._ironpy)
        for release in RELEASE_CLASSES.keys():
            def method(r):
                def add_release_of_type(self,release_table=None,outlet=0,name=None,nodes=None):
                    return self.add_release(r,release_table,outlet,name,nodes)
                add_release_of_type.__doc__ = self.add_release.__doc__
                return add_release_of_type
            setattr(self,'add_%s'%release,MethodType(method(release), self))

    def _build_accessor(self, parameter=None, nodes=None):
        return self.node_actions._build_accessor(parameter,nodes=nodes,node_types='StorageNodeModel')

    def load_lva(self,lva_table,nodes=None):
        '''
        Load a Level/Volume/Area table into one or more storage nodes.

        lva_table: A DataFrame with columns level (in metres), volume (in m^3), and area (in m^2)

        nodes: Option node name or list of node names to apply LVA too. Default: apply to all storage nodes
        '''
        lva_text = '\n'.join(['geo.Add(%f,%f,%f)'%(r['level'],r['area'],r['volume']) for _,r  in lva_table.iterrows()])
        code = LOAD_LVA_SCRIPTLET%lva_text
        return self.node_actions.apply(code,init='0',node_types='StorageNodeModel',nodes=nodes)

    def add_release(self,release_type,release_table=None,outlet=0,name=None,nodes=None):
        '''
        Create a new release on one or more storage nodes

        release_table: if provided, should be a DataFrame with columns level (in m), minimum (in m^3.s^-1) and maximum (in m^3.s^-1)
        outlet: if an integer, apply release to the outlet path in that position on each storage.
                if a string, apply release to that downstream link (which will only work if you are only applying to one storage node)

        name: name for new release. If not provided, will be '<releasetype> from script'.
              IF the name already exists on another release on a particular storage, the existing release will be replaced!
        '''
        if not release_type in RELEASE_CLASSES:
            raise Exception("Unknown release type: %s"%release_type)
        if name is None:
            name = '"%s from script"'%release_type

        klass = RELEASE_CLASSES[release_type]
        get_outlet = path_query(outlet)
        code = ADD_RELEASE_SCRIPTLET%(klass,get_outlet,klass,name)

        def setup_release_curve(tbl,col):
            return '\n'.join(['new_release.%sRelease.Add(%f,%f)'%(col.capitalize(),r["level"],r[col]) for _,r in tbl.iterrows()])
        if release_table is not None:
            setup_min = setup_release_curve(release_table,'minimum')
            setup_max = setup_release_curve(release_table,'maximum')
            code = '\n'.join([code,setup_min,setup_max])

        return self.node_actions.apply(code,init='0',node_types='StorageNodeModel',nodes=nodes)

    def outlets(self,nodes=None):
        return self.get_param_values('OutletPaths.*Link.Name',nodes=nodes)

    def lva(self,node):
        '''
        Retrieve the Level/Volume/Area table for a given storage node
        '''
        code = GET_LVA_SCRIPTLET
        res = self.apply(code,init='[]',nodes=[node])
        vals = [eval(r) for r in res]
        
        return pd.DataFrame(vals,columns=['level','area','volume'])

    def releases(self,nodes=None,path=None):
        path_q = path_query(path)
        if path_q == '':
            indexer = '.*'
        else:
            indexer = '.'

        return self.get_param_values('ProductReleaseContainer.Releases%s%sReleaseItemName'%(path_q,indexer),nodes=nodes)

    def release_table(self,node,release):
        code = GET_RELEASE_TABLE_SCRIPTLET%release
        res = self.apply(code,init='[]',nodes=[node])

        df = pd.DataFrame(res,columns=['level','minimum','level_from_max','maximum'])
        if not np.all(df.level==df.level_from_max):
            raise Exception('Inconsistent levels between minimum and maximum release curves')
        return df[['level','minimum','maximum']]



