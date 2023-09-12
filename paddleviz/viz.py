import os
import paddle

from graphviz import Digraph


# The map of grad_nodes, which key is point of operator and value is point of grad_node
grad_nodes = {}

def make_graph(var):
    """visualize reversed graph

    :param var: output of the network's forward process
    :return dot: result of reversed graph, its type is `graphviz.Digraph`
    """
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12", rankdir="BT", dpi="600"), strict=True)
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


def processOPLog(str, op_id, dot):
  """
  str: the log of operator
  op_id: the pointer of oparator
  dot: the whole dot
  """
  start = str.find("Input")

  while str[start] != '\n':
    start += 1

  start = str.find('(', start) + 1

  end = str.find("Output")

  # handle node of input
  p = start

  # each line of the operator's input is processed in turn using a double pointer
  while p != -1 and p < end:

    # find carriage return
    while str[p] != '\n':
      p += 1

    # find the properties of the edge
    name = str[start + 1: str.find(',', start)].strip(' ')
    
    start = str.find("Ptr", start)
    ptr = str[start + 5: str.find(',', start)]

    start = str.find("Dtype", start)
    dtype = str[start + 7: str.find(',', start)]

    start = str.find("Place", start)
    place = str[start + 7: str.find(',', start)]

    start = str.find("Shape", start)
    shape = '[{}]'.format(str[start + 7: str.find(']', start)].strip(' '))

    p = str.find('(', p, end)

    start = p
    
    if 'grad' not in name:
      continue

    print("input_op: {} -> {}".format(op_id, ptr))

    if ptr not in grad_nodes:
      grad_nodes[ptr] = {}

    # if have previously recorded which operator output came from, need to add information on the side
    if "output_op" in grad_nodes[ptr]:
      edge_info = "dtype: {} \n place: {} \n shape: {} \n".format(dtype, place, shape)
      dot.edge(grad_nodes[ptr]["output_op"], op_id, _attributes={'label': edge_info})
    
    

  # handle node of output
  start = end
  while str[start] != '\n':
    start += 1

  start = str.find('(', start)

  end = len(str)
  
  p = start

  # each line of the operator's output is processed in turn using a double pointer
  while p != -1 and p < end:

    # find carriage return
    while p < end and str[p] != '\n':
      p += 1

    # find the properties of the edge
    name = str[start + 1: str.find(',', start)].strip(' ')

    start = str.find("Ptr", start)
    ptr = str[start + 5: str.find(',', start)]

    if ptr not in grad_nodes:
      grad_nodes[ptr] = {}
    
    grad_nodes[ptr]["output_op"] = op_id
    print("output_op: {} -> {}".format(op_id, ptr))

    p = str.find('(', p, end)
  
    start = p


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
    op_ptr = content[start: end]
    
    end = content.find("backward.cc:288", start)

    op_log = content[start: end]

    processOPLog(op_log, op_ptr, dot)