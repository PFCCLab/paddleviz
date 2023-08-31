import json

# grad_nodes 的map集合，key 为梯度算子的指针
grad_nodes = {}


def processOPLog(str, op_id, dot):
  """
  str: 该op的输入输出日志
  op_id: 该op的指针
  dot: 整张图 
  """
  start = str.find("Input")
  end = str.find("Output")

  # 处理Input中的grad_node，Input中的grad之前已经被创建过了
  while str.find("grad_", start, end) != -1 :
    
    start = str.find("grad_", start)
    name = str[start: str.find(',', start)]
    
    start = str.find("Ptr", start)
    ptr = str[start + 5: str.find('TensorInfo', start) - 1]

    start = str.find("Dtype", start)
    dtype = str[start + 7: str.find(',', start)]

    start = str.find("Place", start)
    place = str[start + 7: str.find(',', start)]

    start = str.find("Shape", start)
    shape = '[{}]'.format(str[start + 7: str.find(']', start)].strip(' '))

    if ptr not in grad_nodes:
      grad_nodes[ptr] = {}

    if "output_op" in grad_nodes[ptr]:
      edge_info = "ptr: {} \n dtype: {} \n place: {} \n shape: {} \n".format(ptr, dtype, place, shape)
      # dot.remove_edge(str(input_op), grad_node["output_op"])
      dot.edge(op_id, grad_nodes[ptr]["output_op"], _attributes={'label': edge_info})

      # grad_nodes[ptr]["input_op"] += [op_id]
    # else:
    #   grad_nodes[ptr]["input_op"] = [op_id]

  # 处理Output中的grad_node
  while str.find("grad_", start) != -1: 
    
    start = str.find("grad_", start) + 1

    if ptr not in grad_nodes:
      grad_nodes[ptr] = {}
    
    
    grad_nodes[ptr]["output_op"] = op_id


def add_edge_info(dot):
  with open('C:\\Users\\25942\\Desktop\\output.txt', encoding='utf-8') as f:
    content = f.read()
  
  start = 0

  while content.find('gradnode_ptr', start) != -1:
    start = content.find('gradnode_ptr', start)
    end = content.find('\n', start)
    if content[end - 1] == '\r':
      end -= 1
    
    start += 14
    op_ptr = content[start: end]
    
    end = content.find("backward.cc:288", start)

    op_log = content[start: end]

    processOPLog(op_log, op_ptr, dot)


# from graphviz import Digraph
# dot = Digraph()
# add_edge_info(dot)