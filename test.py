from extractor import convert
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import unittest
import subprocess

def exec_test(code):
    # Run Python code safely using subprocess
    codeproc = subprocess.run(
        ["python", "-c", code], 
        stdout=subprocess.PIPE,
        text=True
    )
    return codeproc.stdout.strip()

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class TestMethodExtractor(unittest.TestCase):
    def helper_test_code_snippet(self, code):
        code = code.encode("utf-8")
        tree = parser.parse(code)
        lines = len(tree.root_node.children)
        original_result = exec_test(code)
        for end in range(0, lines - 1): # exclude print statement
            for start in range(0, end + 1):
                tree = parser.parse(code)
                result = convert(tree.root_node, start, end, parser)
                converted_result = exec_test(result.text)
                self.assertEqual(
                    original_result,
                    converted_result,
                    f"Error while extracting {start}, {end}: got {converted_result} instead of {original_result}"
                )
    def test_basic(self):
        self.helper_test_code_snippet('''
x = 1
y = 2
x = y + 1
y = x + 2
print(x, y)
        ''')
    
    def test_multiple_assignment(self):
        self.helper_test_code_snippet('''
x, y = 1, 2
y, z = 3, 4
x, y = y, (x + 1)
y, z = z, y
z, x = x, y                    
print(x, y, z)
        ''')

    def test_lambda(self):
        self.helper_test_code_snippet('''
x = 1
inc = lambda x: (x + 1)
y = inc(2)
z = inc(5 + x)
print(x, y, z)                                  
        ''')

if __name__ == "__main__":
    unittest.main()

