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
    'LBRACKET', 'RBRACKET', 'COMMA', 'COLON', 'SEMICOLON', 'DOT','ADD','SUB','MULT','DIV',
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


# Grammar rules
def p_prog(p):
    '''prog : buildClassOrFunc'''
    pass


def p_buildClassOrFunc(p):
    '''buildClassOrFunc : classDecl
                       | funcDef'''
    pass

def p_classDecl(p):
    '''classDecl : CLASS ID LBRACE RBRACE
                 | CLASS ID LBRACE memberDecl RBRACE'''
    pass

def p_memberDecl(p):
    '''memberDecl : memberFuncDecl
                  | memberVarDecl
                  | memberDecl memberDecl'''
    pass

def p_memberFuncDecl(p):
    '''memberFuncDecl : FUNCTION ID COLON LPAREN fParams RPAREN ARROW returnType
                    | CONSTRUCTOR COLON LPAREN fParams RPAREN
                    | FUNCTION ID COLON LPAREN RPAREN ARROW returnType
                    | CONSTRUCTOR COLON LPAREN RPAREN'''
    pass

def p_memberVarDecl(p):
    '''memberVarDecl : TYPE ID COLON TYPE SEMICOLON
                    | TYPE ID COLON TYPE LBRACKET INTLIT RBRACKET SEMICOLON'''
    pass

def p_funcDef(p):
    '''funcDef : funcHead funcBody'''
    pass

def p_funcHead(p):
    '''funcHead : FUNCTION LPAREN ID RPAREN ARROW returnType
                | FUNCTION LPAREN ID construct RPAREN LPAREN fParams RPAREN ARROW returnType
                | FUNCTION LPAREN RPAREN ARROW returnType
                | FUNCTION LPAREN ID construct RPAREN LPAREN RPAREN ARROW returnType'''
    pass

def p_construct(p):
    '''construct :'''
    pass

def p_funcBody(p):
    '''funcBody : LBRACE RBRACE
                | LBRACE statement RBRACE'''
    pass

def p_statement(p):
    '''statement : assignStat
                 | ifStat
                 | whileStat
                 | readStat
                 | writeStat
                 | returnStat
                 | functionCall'''
    pass

def p_assignStat(p):
    '''assignStat : variable assignOp expr'''
    pass


def p_ifStat(p):
    '''ifStat : IF LPAREN relExpr RPAREN THEN statBlock ELSE statBlock'''
    pass


def p_whileStat(p):
    '''whileStat : WHILE LPAREN relExpr RPAREN statBlock'''
    pass


def p_readStat(p):
    '''readStat : READ LPAREN variable RPAREN'''
    pass


def p_writeStat(p):
    '''writeStat : WRITE LPAREN expr RPAREN'''
    pass


def p_returnStat(p):
    '''returnStat : RETURN LPAREN expr RPAREN'''
    pass


def p_functionCall(p):
    '''functionCall : ID LPAREN aParams RPAREN'''
    pass


def p_statBlock(p):
    '''statBlock : LBRACE statement RBRACE'''
    pass


def p_expr(p):
    '''expr : arithExpr
            | relExpr'''
    pass


def p_relExpr(p):
    '''relExpr : arithExpr relOp arithExpr'''
    pass


def p_arithExpr(p):
    '''arithExpr : term addOp arithExpr
                 | term'''
    pass


def p_term(p):
    '''term : factor multOp term
            | factor'''
    pass


def p_factor(p):
    '''factor : variable
              | functionCall
              | INTLIT
              | FLOATLIT
              | LPAREN arithExpr RPAREN
              | NOT factor
              | SIGN factor'''
    pass


def p_variable(p):
    '''variable : ID idnest
                | ID idnest indice'''
    pass


def p_idnest(p):
    '''idnest : DOT ID
              | DOT ID idnest'''
    pass


def p_indice(p):
    '''indice : LBRACKET arithExpr RBRACKET'''
    pass


def p_arraySize(p):
    '''arraySize : LBRACKET INTLIT RBRACKET
                 | LBRACKET RBRACKET'''
    pass


def p_type(p):
    '''type : TYPE
            | VOID'''
    pass


def p_returnType(p):
    '''returnType : type
                  | VOID'''
    pass


def p_fParams(p):
    '''fParams : ID COLON type arraySize COMMA fParams
               |'''
    pass

def p_aParams(p):
    '''aParams : expr COMMA aParams
               |'''
    pass


def p_assignOp(p):
    '''assignOp : EQ'''
    pass


def p_relOp(p):
    '''relOp : EQ
             | NEQ
             | LT
             | GT
             | LE'''
    pass


def p_addOp(p):
    '''addOp : ADD
             | SUB'''
    pass


def p_multOp(p):
    '''multOp : MULT
              | DIV'''
    pass


# Error rule for syntax errors
def p_error(p):
    if p:
        print(f"Syntax error at line {p.lineno}")
    else:
        print("Syntax error: Unexpected end of input")



# Build the parser
parser = yacc.yacc()

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
    return tokens

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
            else:
                tokens.append((tok.string, '50', tok.start[0]))
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

