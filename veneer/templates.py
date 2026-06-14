
APPLY_FUNCTION_INIT='''
import RiverSystem.Functions.FunctionUsage as FunctionUsage
import TIME.Tools.Reflection.ReflectedItem as ReflectedItem

orig_fn_names = %s
def search_by_name(nm):
  if '.' in nm:
    return lambda f: f.FullName==nm
  return lambda f: f.Name==nm

orig_fns = [scenario.Network.FunctionManager.Functions.Where(search_by_name(ofn)).FirstOrDefault() for ofn in orig_fn_names]
for i,fn in enumerate(orig_fns):
  if fn is None:
    raise Exception('Unknown function: '+orig_fn_names[i])
functions = orig_fns[::-1]
'''

APPLY_FUNCTION_LOOP='''
ignoreExceptions=True
assert target is not None
ignoreExceptions=False

if not len(functions):
    functions = orig_fns[::-1]
try:
  target = target%s

  ri = ReflectedItem.NewItem('%s',target)

  existing_usage = scenario.Network.FunctionManager.GetFunctionUsage(ri)
  if existing_usage is not None:
    scenario.Network.FunctionManager.RemoveUsage(ri)

  fn = functions.pop()
  usage = FunctionUsage()
  usage.ReflectedItem = ri

  fn.Usages.Add(usage)
  result["success"] += 1
except:
  result["fail"] += 1
  raise
'''

CLEAR_FUNCTION_LOOP='''
import TIME.Tools.Reflection.ReflectedItem as ReflectedItem

try:
  target = target%s
  ri = ReflectedItem.NewItem('%s',target)
  existing_usage = scenario.Network.FunctionManager.GetFunctionUsage(ri)
  if existing_usage is not None:
    scenario.Network.FunctionManager.RemoveUsage(ri)
  result["success"] += 1
except:
  result["fail"] += 1
  raise
'''

CLEAR_TIMESERIES_LOOP='''
import TIME.Tools.Reflection.ReflectedItem as ReflectedItem

target = %s__init__.__self__
nested = '%s'.split('.')
prop = nested[-1]
for n in nested[:-1]:
  target = getattr(target,n)
ri = ReflectedItem.NewItem(prop,target)
dm = scenario.Network.DataManager
if dm.IsUsed(ri):
  dm.RemoveUsage(ri)
'''

BUILD_PVR_LOOKUP='''
pvt_lookup = {}
for pvr in scenario.ProjectViewTable():
  if not pvr.ObjectReference in pvt_lookup:
    pvt_lookup[pvr.ObjectReference] = []
  pvt_lookup[pvr.ObjectReference].append(pvr)

'''

ENUM_PVRS='''
ignoreExceptions=False
pvrs = pvt_lookup.get(target.__init__.__self__,[])
pvrs = [(str(pvr),str(pvr.ElementRecorder.__class__),pvr.ElementName,[a.KeyString for a in pvr.ElementRecorder.RecordableAttributes]) for pvr in pvrs if System.String.IsNullOrEmpty('%s') or pvr.ElementName.StartsWith('%s')]
if len(pvrs):result.append(pvrs)
'''

CREATED_MODELLED_VARIABLE='''
ignoreExceptions=False
if not len(names): names = orig_names[::-1]
pvrs = pvt_lookup.get(target.__init__.__self__,[])
element_name = '%s'
attribute_name = '%s'
variable_name = '%s'
pvrs = [pvr for pvr in pvrs if System.String.IsNullOrEmpty(attribute_name) or pvr.ElementName.StartsWith(element_name)]
match=None
var_name = names.pop()
name_to_use = variable_name if len(variable_name) else var_name

if len(pvrs)==1:
    # Nested
    #print('Found one PVR')
    match = pvrs[0]
elif len(pvrs)>1:
    matches = [pvr for pvr in pvrs if pvr.ElementName==element_name]
    if len(matches): match=matches[0]
name_exists = scenario.Network.FunctionManager.Variables.Any(lambda v: v.Name==var_name)
name_valid = valid_identifier(name_to_use)
if match and name_valid and not name_exists:
    mv = ModelledVariable()
    try:
      mv.AttributeRecordingStateName = attribute_name
    except:
      from TIME.ScenarioManagement import RecordableItemTransitionUtil
      strings = match.ElementRecorder.RecordableItems.Select(lambda ri: RecordableItemTransitionUtil.GetLegacyKeyString(ri)).ToList()

      matching_item = match.ElementRecorder.RecordableItems.FirstOrDefault(lambda ri : RecordableItemTransitionUtil.GetLegacyKeyString(ri).endswith(attribute_name))

      if matching_item is None:
        mv.AssignedRecordableItemKey = RecordableItemTransitionUtil.GetKeyForDisplayName(match,attribute_name)
      else:
        mv.AssignedRecordableItemKey = matching_item.Key
    assert match.ElementRecorder
    mv.ProjectViewRow = match
    mv.Name = name_to_use
    current_time_step = scenario.Network.FunctionManager.DateRanges.First(lambda dr: dr.Name=='Current Time Step')
    mv.DateRange = current_time_step
    try:
      mv.Reset(scenario.Network.FunctionManager)
      scenario.Network.FunctionManager.Variables.Add(mv)
      result['created'].append(mv.Name)
      if len(variable_name):
        break
    except Exception as e:
      result['failed'].append(name_to_use + ' ' + e.message + ' on ' + str(match))
else:
    result['failed'].append(name_to_use + ' match:' + str(match) + ', valid: ' +str(name_valid) + ', exists: '+ str(name_exists))
'''

CREATE_TS_VARIABLE_SCRIPT='''
from RiverSystem.Functions.Variables import TimeSeriesVariable

fm = scenario.Network.FunctionManager
grp='%s'
variables = %s
columns = %s
new_variables = []
existing_variables = []
for var_name, column in zip(variables,columns):
    new_var = fm.Variables.FirstOrDefault(lambda v: v.Name==var_name)
    if new_var is None:
        new_var = TimeSeriesVariable()
        new_var.Name = var_name
        new_variables.append(var_name)
    else:
        existing_variables.append(var_name)

    H.AssignTimeSeries(scenario,new_var,'Value',grp,column)
    fm.Variables.Add(new_var)
result = {
    'created':new_variables,
    'updated':existing_variables
}
'''

CREATE_PIECEWISE_VARIABLE_SCRIPT='''
from RiverSystem.Functions.Variables import LinearVariable, LinearFunctionVariableEntry

fm = scenario.Network.FunctionManager
variable = '%s'
x_name = '%s'
y_name = '%s'
new_variable = []
existing_variable = []
new_var = LinearVariable()

e = LinearFunctionVariableEntry()
e.X = 0
e.Y = 1
new_var.Entries.Add(e)
e = LinearFunctionVariableEntry()
e.X = 1
e.Y = 1
new_var.Entries.Add(e)
new_var.Name = variable
new_var.XName = x_name
new_var.YName = y_name
fm.Variables.Add(new_var)
result = True
'''

VALID_IDENTIFIER_FN='''
def valid_identifier(nm):
  from System.Text import RegularExpressions

  if not nm:
    return False
  if len(nm) < 2:
    return False
  if nm[0] != '$':
    return False

  return RegularExpressions.Regex("[_A-Za-z][_a-zA-Z0-9]*$").IsMatch(nm[1:])
'''

CREATE_FUNCTIONS='''
import RiverSystem.Functions.Function as Function
import RiverSystem.DataManagement.DataManager.FolderItem as FolderItem
import RiverSystem.Utils.UnitLibrary as UnitLibrary

full_function_path = %s

mgr = scenario.Network.FunctionManager

def full_identifier(nm):
  if full_function_path is None:
    return nm
  if nm.startswith('$'):
    nm = nm[1:]
  return full_function_path + '.' + nm

'''+VALID_IDENTIFIER_FN+'''

functions = %s

parent = None
if full_function_path is not None:
  function_path=None
  for folder in full_function_path.split('.'):
    if function_path is None:
      function_path = folder
    else:
      function_path += '.' + folder
    existing = mgr.Folders.FirstOrDefault(lambda f: f.FullName=="$"+function_path)
    if existing is None:
      new_folder = FolderItem()
      new_folder.Name = folder
      new_folder.Parent = parent
      mgr.Folders.Add(new_folder)
      parent = new_folder
    else:
      parent = existing

result={"created":[],"failed":[]}
for (fn,expr) in functions:
  if not fn.startswith("$"): fn = "$"+fn
  if not valid_identifier(fn):
    result["failed"].append(fn)
    continue
  full_name = full_identifier(fn)
  if mgr.Functions.Any(lambda f: f.FullName==full_name):
    result["failed"].append(fn)
    continue
  rsFn = Function()
  rsFn.Name=fn
  rsFn.Expression=expr
  rsFn.Parent = parent
  scenario.Network.FunctionManager.Functions.Add(rsFn)
  result["created"].append(fn)
scenario.Network.FunctionManager.Refresh()
'''

FIND_MODELLED_VARIABLE_TARGETS='''
ignoreExceptions=False
pvrs = pvt_lookup.get(target.__init__.__self__,[])
#pvrs = [pvr for pvr in pvrs if System.String.IsNullOrEmpty(attribute_name) or pvr.ElementName.StartsWith(element_name)]

for pvr in pvrs:
  if not pvr.ElementName in result:
    result[pvr.ElementName] = []
  for attr in pvr.ElementRecorder.RecordableAttributes:
    result[pvr.ElementName].append(attr.KeyString)
'''


GET_TIME_PERIODS='''
from RiverSystem.Management.ExpressionBuilder import DateRangeListUtility
#result = DateRangeListUtility.DateRanges().Select(lambda dr: dr.Name)
result = scenario.Network.FunctionManager.DateRanges.Select(lambda dr: dr.Name)
'''

SET_TIME_PERIODS='''
from RiverSystem.Management.ExpressionBuilder import DateRangeListUtility
date_range = scenario.Network.FunctionManager.DateRanges.First(lambda dr: dr.Name=='%s')
targets = %s

fm = scenario.Network.FunctionManager
result = []
for variable in fm.Variables:
    if (targets is not None) and variable.Name not in targets and variable.FullName not in targets:
        result.append('%%s not in targets'%%variable.FullName)
        continue
    result.append('Updating %%s to DateRange=%%s'%%(variable.FullName,date_range.Name))
    variable.DateRange = date_range
'''

SET_MODEL_VARIABLE_UNITS='''
import RiverSystem.Management.ExpressionBuilder.TimeOfEvaluation as TimeOfEvaluation
import RiverSystem.Utils.UnitLibrary as UnitLibrary
from TIME.Core import Unit, CommonUnits
targets = %s

new_units=Unit.PredefinedUnit(CommonUnits.%s)

fm = scenario.Network.FunctionManager
result = []
for variable in fm.Variables:
    if (targets is not None) and variable.Name not in targets and variable.FullName not in targets:
        result.append('%%s not in targets'%%variable.FullName)
        continue
    result.append('Updating %%s to ResultUnit=%%s'%%(variable.FullName,new_units))
    variable.ResultUnit = new_units
'''

CREATE_FUNCTIONAL_UNIT='''
import RiverSystem.Catchments.FunctionalUnitDefinition as FunctionalUnitDefinition
import  RiverSystem.Catchments.StandardFunctionalUnit as StandardFunctionalUnit
the_list = scenario.SystemFunctionalUnitConfiguration.fuDefinitions
new_fu = '%s'
fud = None
if the_list.Any(lambda fud: fud.Name==new_fu):
  print('Existing')
  fud = the_list.First(lambda fud:fud.Name==new_fu)
else:
  fud = FunctionalUnitDefinition(new_fu)
  the_list.Add(fud)
  print('Not existing')

for c in scenario.Network.Catchments:
  fu = StandardFunctionalUnit()
  fu.definition = fud
  fu.catchment = c
  c.FunctionalUnits.AddIfNeeded(fu)

H.EnsureElementsHaveConstituentProviders(scenario)
H.InitialiseModelsForConstituentSource(scenario)

'''

SAVE_RASTER_SCRIPT='''
from TIME.Management import NonInteractiveIO
NonInteractiveIO.Save(r"%s",scenario.GeographicData.%s)
'''

TABULATION_SCRIPTLET='''
ignoreExceptions=False
tbl = target.{property_name}
for row in tbl:
    result.append(({values}))
'''

LOAD_PIECEWISE_ROUTING_TABLE_SCRIPTLET = '''
from RiverSystem.Flow import Piecewise
ignoreExceptions=False
pw_table = target.Piecewises
pw_table.Clear()
%s
result += 1
'''

LOAD_RATING_TABLE_SCRIPTLET='''
from RiverSystem.GWSWLink import LinkRatingCurve, LinkRatingCurvePoint
ignoreExceptions=False
curve = LinkRatingCurve()
%s
curve.StartDate = System.DateTime(%s,%s,%s)
curve.OverbankFlowlevel=%f
target.link.RatingCurveLibrary.Curves.Add(curve)
result += 1
'''

SET_INPUT_SET_SCRIPT='''
input_set = scenario.Network.InputSets.First(lambda i: i.Name=="%s")
scenario.CurrentConfiguration.SelectedInputSet = input_set
'''

WIDEST_DATE_RANGE_SCRIPT='''
from TIME.DataTypes import TimeTools, TimeSeries
config = scenario.CurrentConfiguration
input_set = config.SelectedInputSet
temporalCharacteristics = TimeTools.findTemporalCharacteristics(
    scenario.Network.DataManager.GetTimeSeriesForValidation(input_set).ToList[TimeSeries]().ToArray(),
    config.StartDate, config.EndDate);
result = [temporalCharacteristics.minimum,temporalCharacteristics.maximum]
'''

# Namespace import for the recorder set manager (an IPluginDataModel), used by the
# recorder set helpers on VeneerSimulationActions.
RECORDER_SET_MANAGER_NS='RiverSystem.ScenarioSets.RecorderSets.RecorderSetManager as RecorderSetManager'

# Replaces the recorder set selection for the current run configuration. A named
# recorder set, once selected, OVERRIDES v.configure_recording() at run time. An
# empty selection reverts to the special 'Current Recorder Tree' set, ie recording
# is then governed by the manually configured recorders (v.configure_recording()).
#
# Only SelectedRecorderSets is mutated: PersistedSelectedRecorderSets is a derived
# view that follows the selection, so there is nothing separate to update. We also
# avoid reading PersistedSelectedRecorderSets here, as doing so can cause Source to
# inject the 'Current Recorder Tree' set into the selection as a side effect.
#
# Format arg: wanted (a Python list literal of set names).
SELECT_RECORDER_SETS_SCRIPT='''
mgr = scenario.GetPluginModel[RecorderSetManager]()
selected = scenario.CurrentConfiguration.SelectedRecorderSets
wanted = %(wanted)s
available = [s.Name for s in mgr.Sets]
missing = [w for w in wanted if w not in available]
selected.Clear()
for recorder_set in mgr.Sets:
    if recorder_set.Name in wanted:
        selected.Add(recorder_set)
result = {'selected': [s.Name for s in selected], 'missing': missing}
'''

# Sets (or clears) a suffix on the Source main window (WinForms) title for the
# lifetime of the Source session. The original title is cached in
# AppDomain.CurrentDomain so repeated set/clear cycles restore exactly and never
# accumulate suffixes.
#
# Format arg: a Python literal for the suffix text (string repr, or 'None' to clear).
SET_STATUS_INDICATOR_SCRIPT='''
import clr
clr.AddReference("System.Windows.Forms")
from System import Action, AppDomain
from System.Windows.Forms import Application as WFApp

KEY = "VeneerStatusIndicator.OriginalTitle"
TEXT = %s

form = None
for f in WFApp.OpenForms:
    if f.Text and f.Text.startswith("Source "):
        form = f
        break
if form is None:
    raise Exception("Source main window not found in WinForms Application.OpenForms")

def _do():
    cached = AppDomain.CurrentDomain.GetData(KEY)
    if cached is None:
        AppDomain.CurrentDomain.SetData(KEY, form.Text or "")
        cached = form.Text or ""
    if TEXT is None or TEXT == "":
        form.Text = cached
    else:
        form.Text = cached + "  " + TEXT

if form.InvokeRequired:
    form.Invoke(Action(_do))
else:
    _do()
'''

# Shows (or removes) a styled banner Label docked at the top of the Source
# WinForms main form. The Label reference is cached in AppDomain so updates
# and clears find the existing control rather than stacking new ones.
#
# Format args (named): text, bg, fg, height, font_size, bold, font_family
# Each is a Python literal as produced by repr().
SET_STATUS_BANNER_SCRIPT='''
import clr
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")
from System import Action, AppDomain
from System.Windows.Forms import Application as WFApp, Label, DockStyle
from System.Drawing import Color, Font, FontStyle, ContentAlignment, ColorTranslator

KEY = "VeneerStatusBanner.Control"
TEXT = %(text)s
BG = %(bg)s
FG = %(fg)s
HEIGHT = %(height)s
FONT_SIZE = %(font_size)s
BOLD = %(bold)s
FONT_FAMILY = %(font_family)s

form = None
for f in WFApp.OpenForms:
    if f.Text and f.Text.startswith("Source "):
        form = f
        break
if form is None:
    raise Exception("Source main window not found in WinForms Application.OpenForms")

def _parse_color(s):
    s = str(s)
    if s.startswith("#"):
        return ColorTranslator.FromHtml(s)
    return Color.FromName(s)

def _do():
    label = AppDomain.CurrentDomain.GetData(KEY)
    if label is not None:
        try:
            if label.IsDisposed:
                label = None
        except:
            label = None

    if TEXT is None or TEXT == "":
        if label is not None:
            form.Controls.Remove(label)
            label.Dispose()
            AppDomain.CurrentDomain.SetData(KEY, None)
        return

    if label is None:
        label = Label()
        label.Dock = DockStyle.Top
        label.AutoSize = False
        label.TextAlign = ContentAlignment.MiddleCenter
        form.Controls.Add(label)
        AppDomain.CurrentDomain.SetData(KEY, label)

    label.Text = TEXT
    label.Height = HEIGHT
    label.BackColor = _parse_color(BG)
    label.ForeColor = _parse_color(FG)
    style = FontStyle.Bold if BOLD else FontStyle.Regular
    label.Font = Font(FONT_FAMILY, float(FONT_SIZE), style)

if form.InvokeRequired:
    form.Invoke(Action(_do))
else:
    _do()
'''