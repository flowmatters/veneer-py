
APPLY_FUNCTION_INIT='''
import RiverSystem.Functions.FunctionUsage as FunctionUsage
import TIME.Tools.Reflection.ReflectedItem as ReflectedItem

orig_fn_names = %s
orig_fns = [scenario.Network.FunctionManager.Functions.Where(lambda f: f.Name==ofn).FirstOrDefault() for ofn in orig_fn_names]
functions = orig_fns[::-1]
'''

APPLY_FUNCTION_LOOP='''
ignoreExceptions=False
if not len(functions):
    functions = orig_fns[::-1]
try:
  ri = ReflectedItem.NewItem('%s',target)
  fn = functions.pop()
  usage = FunctionUsage()
  usage.ReflectedItem = ri
  fn.Usages.Add(usage)
  result["success"] += 1
except:
  result["fail"] += 1
  raise
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
pvrs = [pvr for pvr in pvrs if System.String.IsNullOrEmpty(attribute_name) or pvr.ElementName.StartsWith(element_name)]
match=None
var_name = names.pop()
if len(pvrs)==1:
    # Nested
    #print('Found one PVR')
    match = pvrs[0]
elif len(pvrs)>1:
    matches = [pvr for pvr in pvrs if pvr.ElementName==element_name]
    if len(matches): match=matches[0]
name_exists = scenario.Network.FunctionManager.Variables.Any(lambda v: v.Name==var_name)
if match and valid_identifier(var_name) and not name_exists:
    mv = ModelledVariable()
    mv.AttributeRecordingStateName = attribute_name
    assert match.ElementRecorder
    assert match.ElementRecorder.RecordableAttributes
    mv.ProjectViewRow = match
    mv.Name = var_name
    mv.DateRange = scenario.Network.FunctionManager.DateRanges[0]
    try:
      mv.Reset()
      scenario.Network.FunctionManager.Variables.Add(mv)
      result['created'].append(mv.Name)
    except Exception as e:
      result['failed'].append(var_name + ' ' + e.message + ' on ' + str(match))
else:
    result['failed'].append(var_name)
'''

VALID_IDENTIFIER_FN='''
def valid_identifier(nm):
  import re
  if not nm:
    return False
  if len(nm) < 2:
    return False
  if nm[0] != '$':
    return False

  return re.match("[_A-Za-z][_a-zA-Z0-9]*$",nm[1:])
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