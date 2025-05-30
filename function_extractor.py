# convertor script
from enum import Enum
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
import numpy as np

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


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

def remove_consecutive_returns(statement):
    last_was_return = False
    new_statement = []
    
    for child in statement:
        if child.type == "return_statement":
            if last_was_return:
                continue  # Skip consecutive return
            last_was_return = True
        else:
            last_was_return = False
        new_statement.append(child)
    return new_statement
    
def create_function(name: str, args: list, statements: list, return_args: list, parser):
    args = [arg.decode("utf-8") if isinstance(arg, bytes) else str(arg) for arg in args]
    return_args = [return_arg.decode("utf-8") if isinstance(return_arg, bytes) else str(return_arg) for return_arg in return_args]
    statements = [stmt.text.decode("utf-8") if stmt is not None else "" for stmt in statements]
    statements = flat_map(lambda statement: statement.split('\n'), statements)
    body = "\n\t".join(statements)  # Ensure proper indentation

    function_template = f"""def {name}({', '.join(args)}):
\t{body}
\t{"" if len(return_args) == 0 else f"return {', '.join(return_args)}"}"""
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


class FunctionExtractor:
    threshold = 10 # minimum number of AST per function body
    counter = 0
    def __init__(self, node: Node, parser):
        if (node.type != "function_definition"):
            raise ValueError("Node type must be function definition, got " + node.type)
        self.parser = parser
        self.name = node.child_by_field_name("name").text.decode("utf-8")
        self.params = node.child_by_field_name("parameters").children[1:-1]
        self.statements = node.child_by_field_name("body").children

    def get_fresh_name(self): # get new function name
        self.counter+=1
        return "_" + self.name + str(self.counter) 

    def walk(self, statement):
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
    
    def extract(self, start: int, end: int, start_extracted: int, end_extracted: int):
        table = {}
        statements = self.statements[start: end + 1].copy()
        for i, statement in enumerate(statements):
            wids, rids = self.walk(statement)
            region = (
                Region.BEFORE if i < start_extracted 
                else Region.AFTER if i > end_extracted 
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
            
        func_name = self.get_fresh_name()
        args = get_args(table)
        results = get_results(table)
        replaced_function = create_function(func_name, args, self.statements[start_extracted: end_extracted + 1], results, self.parser)
        return_function = assign_return(func_name, args, results, self.parser)
        statements[start_extracted: end_extracted + 1] = [replaced_function, return_function]
        return statements
    
    def run(self):
        converted_statements = []
        for i in range(0, len(self.statements), self.threshold):
            length = min(self.threshold, len(self.statements) - i)
            if (length < self.threshold):
                converted_statements = converted_statements + self.statements[i: len(self.statements)]
            else:
                left = i + int(length / 4)
                right = i + int(3 * length / 4)
                converted_statements = converted_statements + self.extract(i, i + self.threshold, left, right)
        self.statements = converted_statements
        return create_function(self.name, self.params, self.statements, [], self.parser)

# Example

code = '''
def test():
    x = 1
    y = 2
    x = y + 1
    y = x + 2
    x = 1
    y = 2
    x = y + 1
    y = x + 2
    x = 1
    y = 2
    x = y + 1
    y = x + 2
    x = 1
    y = 2
    x = y + 1
    y = x + 2
    x = 1
    y = 2
    x = y + 1
    y = x + 2
    return x + y
'''
root = parser.parse(code.encode("utf-8")).root_node.children[0]
extractor = FunctionExtractor(root, parser)
print(extractor.run().text.decode("utf-8"))
