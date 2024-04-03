import ply.yacc as yacc
import ply.lex as lex
import tokenize
from io import BytesIO

# Define the tokens
tokens = (
    'ID', 'INTLIT', 'FLOATLIT', 'EQ', 'NEQ', 'LT', 'GT', 'LE',
    'CLASS', 'CONSTRUCTOR', 'FUNCTION', 'ARROW', 'TYPE', 'VOID',
    'IF', 'THEN', 'ELSE', 'WHILE', 'READ', 'WRITE', 'RETURN',
    'NOT', 'SIGN', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'LBRACKET', 'RBRACKET', 'COMMA', 'COLON', 'SEMICOLON', 'DOT',
    'ADD', 'SUB', 'MULT', 'DIV',
)

# Tokens
t_EQ = r'=='
t_NEQ = r'!='
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_CLASS = r'class'
t_CONSTRUCTOR = r'constructor'
t_FUNCTION = r'function'
t_ARROW = r'->'
t_TYPE = r'int|float|double|char|bool|id'  # Replace 'id' with the actual identifier token
t_VOID = r'void'
t_IF = r'if'
t_THEN = r'then'
t_ELSE = r'else'
t_WHILE = r'while'
t_READ = r'read'
t_WRITE = r'write'
t_RETURN = r'return'
t_NOT = r'not'
t_SIGN = r'[+\-]'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_COLON = r':'
t_SEMICOLON = r';'
t_DOT = r'\.'

# Ignore whitespace
t_ignore = ' \t\n'
#----------------------------------------------------------End of laxical analyzer-------------------------------------------------------------

# Token definitions for ID and INTLIT
def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    return t

# Define a rule for comments
def t_COMMENT(t):
    r'\/\/.*'
    pass

def t_INTLIT(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_FLOATLIT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t


# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()

class Node:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children if children is not None else []

# Grammar rules
def p_prog(p):
    '''prog : buildClassOrFunc'''
    p[0] = Node("Program", children=[p[1]])



def p_buildClassOrFunc(p):
    '''buildClassOrFunc : classDecl
                       | funcDef'''
    p[0] = Node("BuildClassOrFunc", children=[p[1]])


def p_classDecl(p):
    '''classDecl : CLASS ID LBRACE RBRACE
                 | CLASS ID LBRACE memberDecl RBRACE'''
    p[0] = Node("ClassDecl", children=[Node(p[1]), Node(p[2])]+p[4:])


def p_memberDecl(p):
    '''memberDecl : memberFuncDecl
                  | memberVarDecl
                  | memberDecl memberDecl'''
    p[0] = Node("MemberDecl", children=p[1:])


def p_memberFuncDecl(p):
    '''memberFuncDecl : FUNCTION ID COLON LPAREN fParams RPAREN ARROW returnType
                    | CONSTRUCTOR COLON LPAREN fParams RPAREN
                    | FUNCTION ID COLON LPAREN RPAREN ARROW returnType
                    | CONSTRUCTOR COLON LPAREN RPAREN'''
    p[0] = Node("MemberFuncDecl", children=[Node(p[1]), Node(p[2]), Node(p[4]), Node(p[6]), Node(p[8])])


def p_memberVarDecl(p):
    '''memberVarDecl : TYPE ID COLON TYPE SEMICOLON
                    | TYPE ID COLON TYPE LBRACKET INTLIT RBRACKET SEMICOLON'''
    p[0] = Node("MemberVarDecl", children=[Node(p[1]), Node(p[2]), Node(p[4])] + ([Node(p[6])] if len(p) > 7 else []))


def p_funcDef(p):
    '''funcDef : funcHead funcBody'''
    p[0] = Node("FuncDef", children=[p[1], p[2]])


def p_funcHead(p):
    '''funcHead : FUNCTION LPAREN ID RPAREN ARROW returnType
                | FUNCTION LPAREN ID construct RPAREN LPAREN fParams RPAREN ARROW returnType
                | FUNCTION LPAREN RPAREN ARROW returnType
                | FUNCTION LPAREN ID construct RPAREN LPAREN RPAREN ARROW returnType'''
    p[0] = Node("FuncHead", children=[Node(p[i]) for i in range(1, len(p))])


def p_construct(p):
    '''construct :'''
    p[0] = Node("Construct")


def p_funcBody(p):
    '''funcBody : LBRACE RBRACE
                | LBRACE statement RBRACE'''
    p[0] = Node("FuncBody", children=[p[2]] if len(p) > 3 else [])


def p_statement(p):
    '''statement : assignStat
                 | ifStat
                 | whileStat
                 | readStat
                 | writeStat
                 | returnStat
                 | functionCall'''
    p[0] = Node("Statement", children=[p[1]])


def p_assignStat(p):
    '''assignStat : variable assignOp expr'''
    p[0] = Node("AssignStat", children=[p[1], Node(p[2]), p[3]])



def p_ifStat(p):
    '''ifStat : IF LPAREN relExpr RPAREN THEN statBlock ELSE statBlock'''
    p[0] = Node("IfStat", children=[p[3], p[6], p[8]])



def p_whileStat(p):
    '''whileStat : WHILE LPAREN relExpr RPAREN statBlock'''
    p[0] = Node("WhileStat", children=[p[3], p[5]])



def p_readStat(p):
    '''readStat : READ LPAREN variable RPAREN'''
    p[0] = Node("ReadStat", children=[p[3]])



def p_writeStat(p):
    '''writeStat : WRITE LPAREN expr RPAREN'''
    p[0] = Node("WriteStat", children=[p[3]])



def p_returnStat(p):
    '''returnStat : RETURN LPAREN expr RPAREN'''
    p[0] = Node("ReturnStat", children=[p[3]])



def p_functionCall(p):
    '''functionCall : ID LPAREN aParams RPAREN'''
    p[0] = Node("FunctionCall", children=[Node(p[1]), p[3]])



def p_statBlock(p):
    '''statBlock : LBRACE statement RBRACE'''
    p[0] = Node("StatBlock", children=[p[2]])



def p_expr(p):
    '''expr : arithExpr
            | relExpr'''
    p[0] = Node("Expr", children=[p[1]])



def p_relExpr(p):
    '''relExpr : arithExpr relOp arithExpr'''
    p[0] = Node("RelExpr", children=[p[1], Node(p[2]), p[3]])



def p_arithExpr(p):
    '''arithExpr : term addOp arithExpr
                 | term'''
    p[0] = Node("ArithExpr", children=[p[1]] + ([Node(p[2]), p[3]] if len(p) > 2 else []))



def p_term(p):
    '''term : factor multOp term
            | factor'''
    p[0] = Node("Term", children=[p[1]] + ([Node(p[2]), p[3]] if len(p) > 2 else []))



def p_factor(p):
    '''factor : variable
              | functionCall
              | INTLIT
              | FLOATLIT
              | LPAREN arithExpr RPAREN
              | NOT factor
              | SIGN factor'''
    p[0] = Node("Factor", children=[p[1]] + ([Node(p[2]), p[3]] if len(p) > 2 else []))



def p_variable(p):
    '''variable : ID idnest
                | ID idnest indice'''
    p[0] = Node("Variable", children=[Node(p[1])] + p[2:])



def p_idnest(p):
    '''idnest : DOT ID
              | DOT ID idnest'''
    p[0] = Node("IdNest", children=[Node(p[2])] + p[3:])



def p_indice(p):
    '''indice : LBRACKET arithExpr RBRACKET'''
    p[0] = Node("Indice", children=[p[2]])



def p_arraySize(p):
    '''arraySize : LBRACKET INTLIT RBRACKET
                 | LBRACKET RBRACKET'''
    p[0] = Node("ArraySize", children=[Node(p[2])] if len(p) > 2 else [])



def p_type(p):
    '''type : TYPE
            | VOID'''
    p[0] = Node("Type", children=[Node(p[1])])



def p_returnType(p):
    '''returnType : type
                  | VOID'''
    p[0] = Node("ReturnType", children=[p[1]])



def p_fParams(p):
    '''fParams : ID COLON type arraySize COMMA fParams
               |'''
    p[0] = Node("FParams", children=[Node(p[1]), Node(p[3]), Node(p[4]), p[6]] if len(p) > 1 else [])


def p_aParams(p):
    '''aParams : expr COMMA aParams
               |'''
    p[0] = Node("AParams", children=[p[1]] + p[3] if len(p) > 1 else [])



def p_assignOp(p):
    '''assignOp : EQ'''
    p[0] = Node("AssignOp", children=[Node(p[1])])



def p_relOp(p):
    '''relOp : EQ
             | NEQ
             | LT
             | GT
             | LE'''
    p[0] = Node("RelOp", children=[Node(p[1])])



def p_addOp(p):
    '''addOp : ADD
             | SUB'''
    p[0] = Node("AddOp", children=[Node(p[1])])



def p_multOp(p):
    '''multOp : MULT
              | DIV'''
    p[0] = Node("MultOp", children=[Node(p[1])])



# Error rule for syntax errors
def p_error(p):
    if p:
        print(f"Syntax error at line {p.lineno}")
    else:
        print("Syntax error: Unexpected end of input")



def extract_tokens(code):
    tokens = []
    for tok in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline):
        if tok.string in ['+', '-', '*', '/', '%']:
            tokens.append((tok.string, 'Arithmetic OP', tok.start[0]))
        elif tok.string in ['==', '!=', '>', '<', '%']:
            tokens.append((tok.string, 'Relational OP', tok.start[0]))
        elif tok.string in ['=', '+=', '-=']:
            tokens.append((tok.string, 'Assignment OP', tok.start[0]))
        elif tok.string in ['(', ')', '{', '}', '[', ']']:
            tokens.append((tok.string, 'Bracket', tok.start[0]))
        elif tok.string in [',', ';', ':', '#', '@', '"']:
            tokens.append((tok.string, 'Punctuator', tok.start[0]))
        elif tok.type == tokenize.NUMBER:
            tokens.append((tok.string, 'Numeral', tok.start[0]))
        elif tok.type == tokenize.NAME:
            if tok.string in ['if', 'else', 'return', 'int', 'for', 'switch', 'case', 'while', 'do', 'float', 'double',
                              'string', 'char']:
                tokens.append((tok.string, 'Keyword', tok.start[0]))
            else:
                tokens.append((tok.string, 'Identifier', tok.start[0]))
        elif tok.type == tokenize.COMMENT:
            tokens.append((tok.string, 'Comment', tok.start[0]))
    return tokens

# Build the parser
parser = yacc.yacc()
def parse(code):
    return parser.parse(code)

def display_tokens(code):
    tokens = extract_tokens(code)
    for tok in tokens:
        print(f"{tok[0]:<15} {tok[1]:<15} {tok[2]:<15}")

def generate(code):
    tokens = []
    for tok in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline):
        if tok.type == tokenize.NUMBER:
            print((tok.string, tok.type, tok.start[0]))
        elif tok.string in ['for', 'while', 'do']:
            print((tok.string, 'Loop', tok.start[0]))
        elif tok.string in ['if', 'else', 'elseif', 'elif']:
            print((tok.string, 'Condition', tok.start[0]))
        elif tok.string in ['int', 'float', 'double', 'char', 'bool']:
            print((tok.string, 'datatype', tok.start[0]))
        elif tok.type == tokenize.NAME:
            if tok.string in ['if', 'else', 'return', 'for', 'switch', 'case', 'while', 'do', 'string', 'scanf', 'input', 'cin', 'printf', 'print', 'cout']:
                print((tok.string, 'Keyword', tok.start[0]))
    return tokens

def generate_table(code):
    tokens = generate(code)
    for tok in tokens:
        print(f"{tok[0]:<15} {tok[1]:<15}")

if __name__ == "__main__":
    code_input = input("Enter your code: ")

    print("\nTokens:")
    display_tokens(code_input)

    print("\nGenerated Table:")
    generate_table(code_input)

    parser.parse(code_input)

    syntax_tree = parse(code_input)
    print("\nSyntax Tree:")
    print(syntax_tree)


