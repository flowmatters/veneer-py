from types import MethodType

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

class VeneerStorageActions(object):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        for release in RELEASE_CLASSES.keys():
            def method(r):
                def add_release_of_type(self,release_table=None,outlet=0,name=None,nodes=None):
                    return self.add_release(r,release_table,outlet,name,nodes)
                add_release_of_type.__doc__ = self.add_release.__doc__
                return add_release_of_type
            setattr(self,'add_%s'%release,MethodType(method(release), self))

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
        if isinstance(outlet,int):
            get_outlet = '[%d]'%outlet
        else:
            get_outlet = '.FirstOrDefault(lambda op:op.Link.Name=="%s")'%outlet
        code = ADD_RELEASE_SCRIPTLET%(klass,get_outlet,klass,name)

        def setup_release_curve(tbl,col):
            return '\n'.join(['new_release.%sRelease.Add(%f,%f)'%(col.capitalize(),r["level"],r[col]) for _,r in tbl.iterrows()])
        if release_table is not None:
            setup_min = setup_release_curve(release_table,'minimum')
            setup_max = setup_release_curve(release_table,'maximum')
            code = '\n'.join([code,setup_min,setup_max])

        return self.node_actions.apply(code,init='0',node_types='StorageNodeModel',nodes=nodes)
