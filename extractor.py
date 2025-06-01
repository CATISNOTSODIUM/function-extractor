from enum import Enum
import numpy as np

def walk(statement):
    walker = statement.walk()
    assigned_variables = []
    identifiers = []
    while True:
        node = walker.node
        if node.type == "assignment":
            if walker.goto_first_child():
                node = walker.node
                if node.type == "identifier":
                    assigned_variables = [walker.node]
                elif node.type == "pattern_list":
                    assigned_variables = [node  for node in walker.node.children if node.type == "identifier"]
            if walker.goto_next_sibling():
                node = walker.node # go to the right side of the assignment
        elif node.type == "identifier":
            identifiers.append(node)
        # Navigate the tree
        if walker.goto_first_child():
            continue
        if walker.goto_next_sibling():
            continue

        # Backtrack when necessary
        while not walker.goto_parent():
            break

        break 
    return assigned_variables, identifiers

class Status(Enum):
    READ = 0
    WRITE_FULL = 1
    WRITE_PARTIAL = 2
    LIVE = 3

    TOTAL = 4

class Region(Enum):
    BEFORE = 0
    WITHIN = 1
    AFTER = 2
    TOTAL = 3


def get_args(table):
    return {k for k, v in table.items() if v[Region.WITHIN.value][Status.LIVE.value] > 0}

def get_results(table):
    return {k for k, v in table.items() if (v[Region.AFTER.value][Status.LIVE.value] > 0)
       and (v[Region.WITHIN.value][Status.WRITE_FULL.value] > 0)} # entry point

def flat_map(f, *iterable):
    import itertools
    return list(itertools.chain.from_iterable(map(f, *iterable)))

def create_function(name: str, args: list, statements: list, return_args: list, parser):
    args = [arg.decode("utf-8") if isinstance(arg, bytes) else str(arg) for arg in args]
    return_args = [return_arg.decode("utf-8") if isinstance(return_arg, bytes) else str(return_arg) for return_arg in return_args]
    statements = [stmt.text.decode("utf-8") if stmt is not None else "" for stmt in statements]
    statements = flat_map(lambda statement: statement.split('\n'), statements)
    body = "\n\t".join(statements)  # Ensure proper indentation

    function_template = f"""def {name}({', '.join(args)}):
\t{body}
\treturn {", ".join(return_args)}"""
    # Convert to bytes for parsing
    function_bytes = function_template.encode("utf-8")
    # Parse the function template
    tree = parser.parse(function_bytes)
    function_node = tree.root_node.children[0]
    # Display the parsed function text
    return function_node


def assign_return(name: str, args: list, return_args: list, parser):
    args = [arg.decode("utf-8") if isinstance(arg, bytes) else str(arg) for arg in args]
    return_args = [return_arg.decode("utf-8") if isinstance(return_arg, bytes) else str(return_arg) for return_arg in return_args]
    if len(return_args) != 0:
        call_template = f"{', '.join(return_args)} = {name}({', '.join(args)})".encode("utf-8")
    else:
        call_template = f"{name}({', '.join(args)})".encode("utf-8")
    tree = parser.parse(call_template)  
    return tree.root_node.children[0]



def extract(statements, start: int, end: int, parser, func_name = "_extracted"):
    # DFS along the possible paths
    stack = []
    stack.append(start)
    table = {}

    while len(stack) > 0:
        cur = stack.pop()
        if (statements[cur].type == "if_statement"):
            print("if statement")

        wids, rids = walk(statements[cur])
        region = (
            Region.BEFORE if cur < start 
            else Region.AFTER if cur > end 
            else Region.WITHIN
        )

        # Check through read before write
        rids_sym = list(map(lambda id: id.text, rids))
        for id_sym in rids_sym:
            if id_sym not in table:
                table[id_sym] = np.zeros([Region.TOTAL.value, Status.TOTAL.value])
            # read
            table[id_sym][region.value][Status.READ.value] = 1
            if (table[id_sym][region.value][Status.WRITE_FULL.value] == 0):
               table[id_sym][region.value][Status.LIVE.value] = 1 
            
        wids_sym = list(map(lambda id: id.text, wids))
        for id_sym in wids_sym:
            if id_sym not in table:
                table[id_sym] = np.zeros([Region.TOTAL.value, Status.TOTAL.value])
            table[id_sym][region.value][Status.WRITE_FULL.value] = 1 # assume always written

        # normal case
        if (cur + 1 <= end):
            stack.append(cur + 1)

    args = get_args(table)
    results = get_results(table)
    replaced_function = create_function(func_name, args, statements[start: end + 1], results, parser)
    return_function = assign_return(func_name, args, results, parser)
    statements[start: end + 1] = [replaced_function, return_function]
    return statements

def convert(root, start: int, end: int, parser):
    statements = root.children
    statements = extract(statements, start, end,  parser)
    statements = [stmt.text.decode("utf-8") if stmt is not None else "" for stmt in statements]
    statements = flat_map(lambda statement: statement.replace('\t','   ').split('\n'), statements)
    body = "\n".join(statements)
    body_bytes = body.encode("utf-8")
    tree = parser.parse(body_bytes)
    return tree.root_node

   


code = '''
x = 2
y = 1
if (x == 1):
    y = 3
else:
    x = 1
'''
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
root = parser.parse(code.encode("utf-8")).root_node
print(root)
print(convert(root, 1, 3, parser).text.decode("utf-8"))