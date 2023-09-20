import os
import paddle

from graphviz import Digraph


# The map of grad_nodes, which key is point of operator and value is point of grad_node
grad_nodes = {}

def make_graph(var, dpi="600"):
    """visualize reversed graph

    :param var: output of the network's forward process
    :param dpi: resolution of graph

    :return dot: result of reversed graph, its type is `graphviz.Digraph`
    """
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12", rankdir="BT", dpi=dpi), strict=True)
    seen = set()

    def add_nodes(fn):
        assert not paddle.is_tensor(fn)
        
        # if already seen, return
        if fn in seen:
            return
        
        # mark node as seen
        seen.add(fn)

        # add the node for this grad_fn
        if fn.name().startswith('GradNodeAccumulation'):
            dot.node(str(hex(fn.node_this_ptr())), fn.name() + '-' + str(hex(fn.node_this_ptr())), fillcolor='#FFCC99', shape='ellipse')
        else:
            dot.node(str(hex(fn.node_this_ptr())), fn.name() + '-' + str(hex(fn.node_this_ptr())), fillcolor='#CCCCFF')
        
        # recurve other nodes
        if hasattr(fn, 'next_functions'):
            # print(fn.name())
            for u in fn.next_functions:
                if u is not None:
                    dot.edge(str(hex(fn.node_this_ptr())), str(hex(u.node_this_ptr())))
                    # print("{}->{}".format(fn.name(), u.name()))
                    add_nodes(u)

    def add_output_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), str(id(var)), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var)), str(hex(var.grad_fn.node_this_ptr())))


    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_output_tensor(v)
    else:
        add_output_tensor(var)


    # add info of edge by log
    add_edge_info(dot)

    # Remove log
    os.remove('./output.txt')
    
    return dot


def add_edge_info(dot):
  with open('./output.txt', encoding='utf-8') as f:
    content = f.read()
  
  start = 0

  while content.find('gradnode_ptr', start) != -1:
    start = content.find('gradnode_ptr', start)
    end = content.find('\n', start)
    if content[end - 1] == '\r':
      end -= 1
    
    start += 15
    op_ptr = content[start: end].strip(' ')
    
    end = content.find("backward.cc:288", start)

    op_log = content[start: end]

    # Get operator input and output objects according to the log
    op_input_output = parseOpLog(op_log, op_ptr, dot)

    
def parseOpLog(op_log, op_ptr, dot):
  """ 
  extract edge info of operator from output log

  :param op_log: the log of operator
  :param op_ptr: the pointer of oparator
  :param dot: the whole dot
  """
  
  op_input_output = {}

  op_log = op_log.replace('\n', ' ')

  start = op_log.find("Input")

  end = op_log.find("Output")

  # Find the input parameters of the operator
  op_input_output["input"] = parseMultiParam(op_log[start: end]) if start != -1 else []

  # Process the input parameters of the operator in turn
  for param in op_input_output["input"]:

    if 'grad' not in param["name"]:
      continue

    param_ptr = param["ptr"]

    if param_ptr not in grad_nodes:
      grad_nodes[param_ptr] = {}

    # if have previously recorded which operator output came from, need to add information on the side
    if "output_op" in grad_nodes[param_ptr]:
      edge_info = "dtype: {} \n place: {} \n shape: {} \n".format(param["dtype"], param["place"], param["shape"])
      dot.edge(grad_nodes[param_ptr]["output_op"], op_ptr, _attributes={'label': edge_info})


  # Find the output parameters of the operator
  op_input_output["output"] = parseMultiParam(op_log[end: ]) if end != -1 else []

  # Process the output parameters of the operator in turn
  for param in op_input_output["output"]:
    param_ptr = param["ptr"]

    if param_ptr not in grad_nodes:
      grad_nodes[param_ptr] = {}
    
    grad_nodes[param_ptr]["output_op"] = op_ptr

  return op_input_output


def parseMultiParam(multi_param_log):
  multi_param = []
  start, end = 0, len(multi_param_log)

  while multi_param_log.find('(', start, end) != -1:
    param_start = multi_param_log.find('(', start, end)
    param_end = multi_param_log.find('}]),', start, end)
    start = param_end + 1
    # Converts a parameter string to an object
    param = parseParam(multi_param_log[param_start + 1: param_end])
    multi_param.append(param)

  return multi_param


def parseParam(param_log):
  param = {}
  start = 0
  
  name = param_log[start: param_log.find(',', start)].strip(' ')
  param["name"] = name
    
  start = param_log.find("Ptr", start)
  ptr = param_log[start + 5: param_log.find(',', start)].strip(' ')
  param["ptr"] = ptr

  start = param_log.find("Dtype", start)
  if start == -1:
    param["dtype"] = None
    param["place"] = None
    param["shape"] = None
    return param
  
  dtype = param_log[start + 7: param_log.find(',', start)].strip(' ')
  param["dtype"] = dtype

  start = param_log.find("Place", start)
  place = param_log[start + 7: param_log.find(',', start)].strip(' ')
  param["place"] = place

  start = param_log.find("Shape", start)
  shape = '[{}]'.format(param_log[start + 7: param_log.find(']', start)].strip(' '))
  param["shape"] = shape

  # print(param)

  return param