import sys
import os
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple
from clang.cindex import Index, CursorKind, TranslationUnit, Config
from collections import defaultdict

from rich.console import Console

Config.set_library_file("/opt/homebrew/Cellar/llvm/18.1.8/lib/libclang.dylib")
console = Console()


# this class will be used during code transformation
@dataclass
class MacroOccurence:
    line: str
    column: str
    file_path: str
    arg_types: List[Tuple[str, str]]
    return_type: str

    def __str__(self):
        arg_types_str = ", ".join(f"({name}: {type_})" for name, type_ in self.arg_types)
        return (
            f"MacroOccurrence:\n"
            f"  File Path: {self.file_path}\n"
            f"  Line: {self.line}, Column: {self.column}\n"
            f"  Argument Types: {arg_types_str}\n"
            f"  Return Type: {self.return_type}\n"
        )
    

@dataclass
class MacroInfo:
    name: str
    is_function_like: bool
    args: List[str]
    body: str
    source_file: str
    line_number: int
    col_number: int
    expressions: Dict[str, List[str]]
    arg_types: Dict[str, Set[str]]
    return_types: Set[str]
    type_occurences: List[MacroOccurence]

class MacroAnalyzer:
    def __init__(self, project_root: str):
        self.index = Index.create()
        self.macros: Dict[str, MacroInfo] = {}
        self.project_root = os.path.abspath(project_root)

        
    def is_project_file(self, filename: str) -> bool:
        """
        Check if a file is part of the project (this is to exclude system headers)
        """
        if not filename:
            return False
            
        abs_path = os.path.abspath(filename)
        
        system_prefixes = [
            "/usr/include",
            "/usr/local/include",
            "/opt/homebrew/include",
            "/Applications/Xcode.app", 
            "C:\\Program Files",        
        ]
        
        # check if file is in system directories
        if any(abs_path.startswith(prefix) for prefix in system_prefixes):
            return False
            

        return abs_path.startswith(self.project_root)

    def get_compilation_args(self, main_file: str):
        """
        Generate compilation arguments including necessary include paths
        """

        args = [
            "-x", "c",      
            "-std=c99",   # use c99 standard
        ]


        args.extend(["-I", self.project_root])
        
        return args

    def analyze_file(self, main_file: str):
        """
        Analyze a C file and its included headers for macros
        """
        args = self.get_compilation_args(main_file)
        
        tu = self.index.parse(
            main_file,
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )
        
        if not tu:
            raise ValueError(f"Unable to parse {main_file}")
        
        self.tu_cursor = tu.cursor
        
        # diagnostics might be important for unexpected ast behavior
        if len(tu.diagnostics) > 0:
            console.print("[red bold] Parse diagnostics:")
            for diag in tu.diagnostics:
                if diag.location.file and self.is_project_file(str(diag.location.file)):
                    console.print(f"[red] {diag.format()}")
        
        self._analyze_tu(tu)

    def _print_tokens(self, tokens):
        print("".join([t.spelling for t in tokens]))

    def add_parens_to_macro_args(self, line, column):
        """
        Adds parantheses around the args to a function-like macro while making sure not to
        add parentheses to args that are macros (see processing on md5.c as an example)
        """

        start_idx = column
        while start_idx < len(line) and line[start_idx] != "(":
            start_idx += 1
        
        if start_idx >= len(line):
            return line  
        
        # find matching closing parenthesis with proper nesting
        paren_count = 1
        end_idx = start_idx + 1
        
        while paren_count > 0 and end_idx < len(line):
            if line[end_idx] == "(":
                paren_count += 1
            elif line[end_idx] == ")":
                paren_count -= 1
            end_idx += 1
        
        if paren_count > 0:
            return line 
        
        # extract macro name and arguments
        macro_name = line[column:start_idx]
        args_str = line[start_idx + 1:end_idx - 1] 
        

        args = []
        current_arg = []
        paren_count = 0
        
        for char in args_str:
            if char == "(" and paren_count == 0 and not current_arg:
                current_arg.append(char)
            elif char == "," and paren_count == 0:
                if current_arg:
                    args.append("".join(current_arg).strip())
                    current_arg = []
            else:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                current_arg.append(char)
        
        if current_arg:
            args.append("".join(current_arg).strip())
        
        # only add parentheses to non macro args
        new_macro = f"{macro_name}({', '.join(['(' + a + ')' if a not in self.macros else a for a in args])})"
        return line[:column] + new_macro + line[end_idx:]
    
    def is_declaration_macro(self, name: str, body: str) -> bool:
        """
        Check if macro is likely to generate declarations (prevent
        adding parentheses that will violate macro declarations)
        """

        declaration_patterns = ["EXTERN", "DECLARE", "TYPEDEF", "STRUCT", "DEFINE_TYPE"]
        
        declaration_keywords = {"extern", "static", "typedef", "struct", "union", "enum", "const", "volatile"}

        type_keywords = {"void", "char", "short", "int", "long", "float", "double",
                        "signed", "unsigned", "bool", "complex", "_Bool", "size_t"}
        
        operators = {"+", "-", "*", "/", "%", "=", "!", "<", ">", "&", "|", "^"}
        
        if any(pattern in name.upper() for pattern in declaration_patterns):
            return True
            
        tokens = body.split()

        if any(keyword in tokens for keyword in declaration_keywords):
            return True
        
        if all(token not in operators and token in type_keywords for token in tokens):
            return True
            
        return False


    def parenthesize_args_and_bodies(self, main_file):
        """
        This will create a new file called  [main_file]_processed.c. It
        adds parentheses around macro bodies and args to function-like macros
        macros passed in as arguments to function-like macros will not be parenthesized
        and declaration like macros will not be parenthesized since they violate C syntax
        """
        # TODO: track which args and bodies have parentheses added to them to help with extra () removal during code transformation
        
        changes = [] # (str, row, col)
        line_to_changes = defaultdict(list)

        
        for c in self.tu_cursor.walk_preorder():
            if c.kind == CursorKind.MACRO_DEFINITION:
                location = c.location
                if location.file and self.is_project_file(str(location.file)):
                    tokens = list(c.get_tokens())
                    if not tokens:
                        continue

                    name_token = tokens[0]
                    name = name_token.spelling
                    
                    body = self._extract_macro_body(tokens[1:])

                    if (self.is_declaration_macro(name, body)):
                        continue


                    if len(tokens) > 1:
                        # get the locations of the name token and potential opening parenthesis
                        name_end = name_token.extent.end.offset
                        next_token = tokens[1]
                        next_start = next_token.extent.start.offset
                        
                        
                        # check if this is a function-like macro
                        if next_token.spelling == "(" and next_start == name_end:
                            args = self._extract_macro_args(tokens[1:])
                            body = self._extract_macro_body(tokens[1:])

                            if not (body.startswith("(") and body.endswith(")")):
                                body = "(" + body + ")"

                            if (args):
                                new_macro = f"#define {name}({','.join(args)}) {body}"
                            else:
                                new_macro = f"#define {name} {body}"
                            
                            changes.append((new_macro, c.location.line, c.location.column))
                            # deal with multi-line macros
                            if (c.location.line != c.extent.end.line):
                                difference = c.extent.end.line - c.location.line
                                for i in range(1, difference + 1):
                                    changes.append(("\n", c.location.line + i, c.location.column))


                        else:
                            # const macro, we can just join all tokens after name
                            body = " ".join(t.spelling for t in tokens[1:])
                            if not (body.startswith("(") and body.endswith(")")):
                                body = "(" + body + ")"
                            new_macro = f"#define {name} {body}"
                            changes.append((new_macro, c.location.line, c.location.column))

                            if (c.location.line != c.extent.end.line):
                                difference = c.extent.end.line - c.location.line
                                for i in range(1, difference + 1):
                                    changes.append(("\n", c.location.line + i, c.location.column))

            if (c.kind == CursorKind.MACRO_INSTANTIATION and 
                c.location.file and 
                self.is_project_file(str(c.location.file))):           
                
                macro_name = c.spelling
                
                if macro_name in self.macros:
                    macro = self.macros[macro_name]
                    
                    if (self.is_declaration_macro(macro.name, macro.body)):
                        continue

                    if macro.is_function_like:
                        exprs = ["(" + e + ")" for e in self._extract_call_expressions(c)]
                        
                        new_macro = f"{macro_name}({', '.join(exprs)})"
                        line_to_changes[c.location.line].append((new_macro, c.location.column))


        base, ext = os.path.splitext(main_file)
        output_file = f"{base}_processed{ext}"

        with open(main_file, "r") as f:
            lines = f.readlines()

        # const macros
        for new_macro, l, col in changes:
            idx = l - 1
            lines[idx] = new_macro + "\n"

        # parenthesize args to function-like macros
        for l in line_to_changes:
            line_changes = line_to_changes[l]
            # IMPORTANT: right to left process order to avoid shifting of location of column
            line_changes = sorted(line_changes, key=lambda x: -x[1], reverse=True)
            idx = l - 1

            current_line = lines[idx]

            for new_macro, col in line_changes:
                current_line = self.add_parens_to_macro_args(current_line, col)
                
            lines[idx] = current_line
        
        with open(output_file, "w") as f:
            f.writelines(lines)
        
     

    def _analyze_tu(self, tu):
        """
        Analyze the translation unit for macros
        """
        self._collect_macro_definitions(tu.cursor)
        self._analyze_macro_usage(tu.cursor)


    def _collect_macro_definitions(self, cursor):
        """
        Collect macro definitions from project files
        """
        for c in cursor.walk_preorder():
            if c.kind == CursorKind.MACRO_DEFINITION:
                location = c.location
                if location.file and self.is_project_file(str(location.file)):
                    tokens = list(c.get_tokens())
                    if not tokens:
                        continue
                    
                    name_token = tokens[0]
                    name = name_token.spelling

                    is_function_like = False
                    args = []
                    body = ""
                    
                    if len(tokens) > 1:
                        # get the locations of the name token and potential opening parenthesis
                        name_end = name_token.extent.end.offset
                        next_token = tokens[1]
                        next_start = next_token.extent.start.offset
                        

                        # check if this is a function-like macro
                        if next_token.spelling == "(" and next_start == name_end:
                            is_function_like = True
                            args = self._extract_macro_args(tokens[1:])
                            body = self._extract_macro_body(tokens[1:])
                        else:
                            # const macro, we can just join all tokens after name
                            body = " ".join(t.spelling for t in tokens[1:])

                    
                    self.macros[name] = MacroInfo(
                        name=name,
                        is_function_like=is_function_like,
                        args=args,
                        body=body,
                        source_file=str(location.file),
                        line_number=location.line,
                        col_number=location.column,
                        expressions={arg: [] for arg in args},
                        return_types=set(),
                        arg_types=dict(),
                        type_occurences=[]
                    )

    def _extract_macro_args(self, tokens) -> List[str]:
        """
        Extract argument names from macro definition, handles nested parentheses
        """
        args = []
        current_arg = []
        paren_count = 0
        in_args = False
        
        for token in tokens:
            spelling = token.spelling
            if spelling == "(":
                if not in_args:
                    in_args = True  
                else:
                    paren_count += 1
                    current_arg.append(spelling)
            elif spelling == ")":
                if paren_count > 0:
                    paren_count -= 1
                    current_arg.append(spelling)
                else:
                    if current_arg:
                        args.append("".join(current_arg).strip())
                    break
            elif spelling == "," and paren_count == 0:
                if current_arg:
                    args.append("".join(current_arg).strip())
                    current_arg = []
            elif in_args:
                current_arg.append(spelling)
        
        return args
    


    def _extract_macro_body(self, tokens) -> str:
        """
        Extract macro body after the parameter list
        """

        body_tokens = []
        paren_count = 0
        param_list_found = False
        in_param_list = False
        
        for token in tokens:
            spelling = token.spelling
            
            # looking for start of parameter list
            if spelling == "(" and not param_list_found:
                param_list_found = True
                in_param_list = True
                continue
                
            if in_param_list:
                if spelling == "(":
                    paren_count += 1
                elif spelling == ")":
                    if paren_count == 0:
                        in_param_list = False  
                        continue
                    paren_count -= 1
            else:
                # after parameter list, collect all tokens
                if not body_tokens: 
                    body_tokens.append(token)
                else:
                    prev_end = body_tokens[-1].extent.end.offset
                    curr_start = token.extent.start.offset
                    if curr_start > prev_end:
                        gap = " " * (curr_start - prev_end)
                        body_tokens.append(gap)
                    body_tokens.append(token)
        
        body = ""
        for token in body_tokens:
            if isinstance(token, str): 
                body += token
            else:
                body += token.spelling
                
        return body.strip()

    def _analyze_macro_usage(self, cursor):
        """
        Analyze macro usage in project files only by looking at
        MACRO_INSTANTIATIONs
        """
        for c in cursor.walk_preorder():
            if (c.kind == CursorKind.MACRO_INSTANTIATION and 
                c.location.file and 
                self.is_project_file(str(c.location.file))):
               
                
                macro_name = c.spelling
                if macro_name in self.macros:
                    macro = self.macros[macro_name]
                    if macro.is_function_like:
                        expressions = self._extract_call_expressions(c)

                        for arg_name, expr in zip(macro.args, expressions):
                            if expr not in macro.expressions[arg_name]:
                                macro.expressions[arg_name].append(expr)
                        
                        
                        analyzed_arg_types = self._analyze_arg_types(c, macro)
                        analyzed_return_type = self._analyze_types(c, macro)
                        macro.type_occurences.append(
                            MacroOccurence(
                                line=c.location.line, column=c.location.column,
                                file_path= os.path.relpath(str(c.location.file), self.project_root),
                                arg_types=analyzed_arg_types,
                                return_type=analyzed_return_type
                                ))
                        
                        
                    else:
                        analyzed_type = self._analyze_types(c, macro)
                        macro.type_occurences.append(
                            MacroOccurence(
                                line=c.location.line, column=c.location.column,
                                file_path= os.path.relpath(str(c.location.file), self.project_root),
                                arg_types=[],
                                return_type=analyzed_type
                                ))


    def _extract_call_expressions(self, cursor) -> List[str]:
        """
        Extract expressions used in macro call
        """

        tokens = list(cursor.get_tokens())
        expressions = []
        current_expr = []
        paren_count = 0
        in_args = False
        
        for token in tokens:
            if token.spelling == "(" and not in_args:
                in_args = True
                continue
            elif token.spelling == "(":
                paren_count += 1
                current_expr.append(token.spelling)
            elif token.spelling == ")":
                if paren_count == 0:
                    if current_expr:
                        expressions.append("".join(current_expr).strip())
                    break
                paren_count -= 1
                current_expr.append(token.spelling)
            elif token.spelling == "," and paren_count == 0:
                if current_expr:
                    expressions.append("".join(current_expr).strip())
                    current_expr = []
            elif in_args:
                current_expr.append(token.spelling)
        
        return expressions
    
    def _get_param_type(self, call_cursor, macro_cursor):
        """
        Get the expected parameter type for a macro used in a function call
        """

        fn = call_cursor.referenced
        if not fn:
            return None
            
        # get the position of the macro in the argument list
        macro_pos = -1
        for i, arg in enumerate(call_cursor.get_arguments()):
            if (arg.extent.start.line == macro_cursor.location.line and
                arg.extent.start.column == macro_cursor.location.column):
                macro_pos = i
                break
                

        if macro_pos >= 0:
            params = list(fn.get_arguments())
            if macro_pos < len(params):
                return params[macro_pos].type.spelling
                
        return None
    

    def _analyze_arg_types(self, cursor, macro: MacroInfo):
        """
        Get the the types of the args passed into a function-like macro
        """

        args_found = []

        for c in self.tu_cursor.walk_preorder():
            sameLocation = c.location.line == cursor.location.line and c.location.column == cursor.location.column
            if (sameLocation and c.kind == CursorKind.PAREN_EXPR):
                # print("-" * 100)
                # print("FOUNDDDDDDDD")
                # print(f"looking at macro {macro.name}")
                # print(c.kind)
                # print(c.type.spelling)
                # start_line, start_col, = c.extent.start.line, c.extent.start.column
                # end_line, end_col = c.extent.end.line, c.extent.end.column
                # print(f"start loc: {start_line}:{start_col}")
                # print(f"end loc: {end_line}:{end_col}")
                found_arg = "".join([c.spelling for c in c.get_tokens()])

                for k, v in self.macros[macro.name].expressions.items():
                    for expr in v:
                        if (expr == found_arg):
                            if k not in macro.arg_types:
                                macro.arg_types[k] = {c.type.spelling}
                            else:
                                macro.arg_types[k].add(c.type.spelling)
                            
                            args_found.append((k, c.type.spelling))
                            
        seen = set()
        return [a for a in args_found if not (a in seen or seen.add(a))]
    

    def _location_contains(self, range_start, range_end, target):
        """
        Check if a location point falls within a source range (inclusive)
        """
        if not (range_start.file and range_end.file and target.file):
            return False
            
        if str(range_start.file) != str(target.file) or str(range_end.file) != str(target.file):
            return False
            
        start = (range_start.line, range_start.column)
        end = (range_end.line, range_end.column)
        point = (target.line, target.column)
        
        return start <= point <= end

    def _find_all_parents(self, target_location, cursor=None, depth=0, parents=None):
        """
        Find all cursors containing the target location, from most specific to least specific.
        """
        if parents is None:
            parents = []
            cursor = self.tu_cursor

        if not cursor:
            return parents

        if self._location_contains(cursor.extent.start, cursor.extent.end, target_location):

            if cursor.kind != CursorKind.MACRO_INSTANTIATION:
                parents.append((cursor, depth, cursor.location))
            
            for child in cursor.get_children():
                self._find_all_parents(target_location, child, depth + 1, parents)

        return parents

    def _analyze_types(self, cursor, macro):
        """
        Analyze the types involved in a macro instantiation by examining its context
        Attempt to, from the instantiation, find the immediate, most enclosing parent cursor
        that gives us type information while skipping over things like compound stmts,
        parens_exprs, unexposed_exprs, etc
        """
        parents = self._find_all_parents(cursor.location)
        # print("inside analyze_types")
 
        #print([parent.kind for parent, _, _ in parents])
        
        if (macro.is_function_like):
            new_parents = []
            for p in parents:
                if p[2].line <= cursor.location.line and p[2].column < cursor.location.column:
                    new_parents.append(p)

            parents = new_parents
   
        # self._print_parent_hierachy(parents)

        type_info = None

        for parent, _, _ in reversed(parents):
            if parent.kind in [
                CursorKind.VAR_DECL,           
                CursorKind.RETURN_STMT,      
            ]:
                # variable declarations and return stmts directly give type
                type_info = parent.type.spelling if parent.type else None
                #print(f"DETERMINED TYPE for macro {macro.name}: {type_info}")
                break
            elif parent.kind == CursorKind.CALL_EXPR:
                # this is when a macro is being passed into a function
                # we can then just get its type from that function's arg type
                type_info = self._get_param_type(parent, cursor)
                #print(f"DETERMINED TYPE for macro {macro.name}: {type_info}")
                break
            elif parent.kind == CursorKind.BINARY_OPERATOR:
                # TODO: need to discuss/learn more about type promotion to verify this is a correct approach
                operands = list(parent.get_children())
                if len(operands) == 2:
                    left_op, right_op = operands
                    macro_loc = cursor.location
    
                    
                    # find sibling operand's type and takes that as its own type
                    if (left_op.extent.start.line == macro_loc.line and 
                        left_op.extent.start.column == macro_loc.column):
                        type_info = right_op.type.spelling
                    elif (right_op.extent.start.line == macro_loc.line and 
                        right_op.extent.start.column == macro_loc.column):
                        type_info = left_op.type.spelling
                    
                    # if type_info:
                    #     print(f"DETERMINED TYPE (binary op) for macro {macro.name}: {type_info}")
                    break
  
            elif parent.kind == CursorKind.FIELD_DECL:
                # struct/union field initialization is just the field's declared type
                type_info = parent.type.spelling
                #print(f"DETERMINED TYPE (field init) for macro {macro.name}: {type_info}")
                break
            elif parent.kind in [CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR, CursorKind.COMPOUND_STMT]:
                continue

        return type_info     
        

    def _print_parent_hierachy(self, parents):
        """
        prints parents of a given cursor in an ast dump format 
        """
        LITERALS = [CursorKind.INTEGER_LITERAL, CursorKind.FLOATING_LITERAL, CursorKind.STRING_LITERAL, CursorKind.CHARACTER_LITERAL, CursorKind.IMAGINARY_LITERAL]
        for parent, depth, location in parents:
            indent = "  " * depth
            prefix = "`-" if depth > 0 else ""
            if parent.kind in LITERALS:
                value = next(parent.get_tokens()).spelling if list(parent.get_tokens()) else ""
                print(f"{indent}{prefix}{parent.kind}: {value} at {location.line}:{location.column}")
            else:
                print(f"{indent}{prefix}{parent.kind}: {parent.spelling} at {location.line}:{location.column}")


        

    def print_analysis(self):
        """
        Print analysis results
        """
        if not self.macros:
            print("\nNo macros found in project files.")
            return
            
        for name, macro in self.macros.items():
            # Get relative path for cleaner output
            rel_path = os.path.relpath(macro.source_file, self.project_root)
            
            print(f"\nMacro: {name}")
            print(f"Defined in: {rel_path}:{macro.line_number}:{macro.col_number}")
            
            if macro.is_function_like:
                print(f"Type: Function-like macro")
                print(f"Arguments: {', '.join(macro.args)}")
                print("\nUsage patterns:")
                for arg_name, expressions in macro.expressions.items():
                    if expressions:
                        print(f"\n{arg_name} used with expressions:")
                        for expr in expressions:
                            print(f"  - {expr}")
                if macro.arg_types:
                    print("arg names and their types")
                    for k, v in macro.arg_types.items():
                        print(f"{k}: {v}")
                if macro.return_types:
                    print("\nObserved return types:")
                    for type_name in macro.return_types:
                        print(f"  - {type_name}")
            else:
                print("Type: Object-like macro")
                if macro.return_types:
                    print("\nObserved return types:")
                    for type_name in macro.return_types:
                        print(f"  - {type_name}")
            
            print(f"\nBody: {macro.body}")
            print("-" * 50)
    
    def print_occurences(self):
        """
        For each macro, print out all of its type occurences
        """

        for macro_name, macro_info in self.macros.items():
            console.print(f"[green bold underline] macro: {macro_name}")
            for o in macro_info.type_occurences:
                print(o)


            print("-"* 50)


def main():
    if len(sys.argv) != 2:
        print("Usage: python macro_analyzer.py <source_file.c>")
        return
    
    # ue the directory of the main file as the root of the project
    main_file = os.path.abspath(sys.argv[1])
    project_root = os.path.dirname(main_file)

    
    analyzer = MacroAnalyzer(project_root)

    try:
        analyzer.analyze_file(main_file)
        analyzer.parenthesize_args_and_bodies(main_file)
        base, ext = os.path.splitext(main_file)


        output_file = f"{base}_processed{ext}"
        project_root = project_root = os.path.dirname(output_file)
        analyzer = MacroAnalyzer(project_root)
        analyzer.analyze_file(output_file)
        #analyzer.print_analysis()
        analyzer.print_occurences()


    except Exception as e:
        console.print(f"[red bold] Error: {e}")

if __name__ == "__main__":
    main()