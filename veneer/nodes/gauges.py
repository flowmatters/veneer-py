import pandas as pd
import numpy as np
from veneer.server_side import VeneerNetworkElementActions


GET_CURVE_SCRIPT=f'''
ignoreExceptions=False
curve = target.RatingCurveLibrary.Curves[%d]
for row in curve.Points:
    result.append((row.Level,row.Discharge,row.Width))
'''

GET_CURVES_SCRIPT=f'''
ignoreExceptions=False
curves = target.RatingCurveLibrary.Curves
for ix,curve in enumerate(curves):
    result.append((ix,curve.Name,curve.StartDate,curve.Points.Count))
'''

LOAD_RATING_SCRIPTLET='''
from RiverSystem.Nodes import RatingCurve, RatingCurvePoint
ignoreExceptions=False
new_table = RatingCurve()
%s
new_table.StartDate = System.DateTime.Parse('%s')
target.RatingCurveLibrary.Curves.Add(new_table)
result += 1
'''

class VeneerGaugeActions(VeneerNetworkElementActions):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        self._name_accessor = self.node_actions._name_accessor
        super(VeneerGaugeActions, self).__init__(node_actions._ironpy)

    def _build_accessor(self, parameter=None, nodes=None):
        return self.node_actions._build_accessor(parameter,nodes=nodes,node_types='GaugeNodeModel')

    def rating_table(self,node,table_name=None,table_index=0):
        if table_name is not None:
            curves = self.rating_tables(node)
            table_index = curves[curves['Name'] == table_name].index[0]
        elif table_index is None:
            raise ValueError("Either table_index or table_name must be provided")

        tbl = self.apply(GET_CURVE_SCRIPT % table_index,init='[]',nodes=[node])
        df = pd.DataFrame(tbl,columns=['Level','Discharge','Width'])
        return df

    def n_rating_tables(self,node):
        return self.get('target.RatingCurveLibrary.Curves.Count',nodes=[node])

    def rating_tables(self,node):
        tbl = self.apply(GET_CURVES_SCRIPT,init='[]',nodes=[node])
        df = pd.DataFrame(tbl,columns=['Index','Name','StartDate','NPoints'])
        return df

    def add_rating_table(self,node,start_date,table):
        rating_text = '\n'.join(['new_table.Points.Add(RatingCurvePoint(Level=%f,Discharge=%f,Width=%f))'%(r['Level'],r['Discharge'],r['Width']) for _,r  in table.iterrows()])
        code = LOAD_RATING_SCRIPTLET%(rating_text,start_date)
        return self.node_actions.apply(code,init='0',node_types='GaugeNodeModel',nodes=[node])
