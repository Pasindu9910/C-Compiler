
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'ADD ARROW CLASS COLON COMMA CONSTRUCTOR DIV DOT ELSE EQ FLOATLIT FUNCTION GT ID IF INTLIT LBRACE LBRACKET LE LPAREN LT MULT NEQ NOT RBRACE RBRACKET READ RETURN RPAREN SEMICOLON SIGN SUB THEN TYPE VOID WHILE WRITEprog : buildClassOrFuncbuildClassOrFunc : classDecl\n                       | funcDefclassDecl : CLASS ID LBRACE RBRACE\n                 | CLASS ID LBRACE memberDecl RBRACEmemberDecl : memberFuncDecl\n                  | memberVarDecl\n                  | memberDecl memberDeclmemberFuncDecl : FUNCTION ID COLON LPAREN fParams RPAREN ARROW returnType\n                    | CONSTRUCTOR COLON LPAREN fParams RPAREN\n                    | FUNCTION ID COLON LPAREN RPAREN ARROW returnType\n                    | CONSTRUCTOR COLON LPAREN RPARENmemberVarDecl : TYPE ID COLON TYPE SEMICOLON\n                    | TYPE ID COLON TYPE LBRACKET INTLIT RBRACKET SEMICOLONfuncDef : funcHead funcBodyfuncHead : FUNCTION LPAREN ID RPAREN ARROW returnType\n                | FUNCTION LPAREN ID construct RPAREN LPAREN fParams RPAREN ARROW returnType\n                | FUNCTION LPAREN RPAREN ARROW returnType\n                | FUNCTION LPAREN ID construct RPAREN LPAREN RPAREN ARROW returnTypeconstruct :funcBody : LBRACE RBRACE\n                | LBRACE statement RBRACEstatement : assignStat\n                 | ifStat\n                 | whileStat\n                 | readStat\n                 | writeStat\n                 | returnStat\n                 | functionCallassignStat : variable assignOp exprifStat : IF LPAREN relExpr RPAREN THEN statBlock ELSE statBlockwhileStat : WHILE LPAREN relExpr RPAREN statBlockreadStat : READ LPAREN variable RPARENwriteStat : WRITE LPAREN expr RPARENreturnStat : RETURN LPAREN expr RPARENfunctionCall : ID LPAREN aParams RPARENstatBlock : LBRACE statement RBRACEexpr : arithExpr\n            | relExprrelExpr : arithExpr relOp arithExprarithExpr : term addOp arithExpr\n                 | termterm : factor multOp term\n            | factorfactor : variable\n              | functionCall\n              | INTLIT\n              | FLOATLIT\n              | LPAREN arithExpr RPAREN\n              | NOT factor\n              | SIGN factorvariable : ID idnest\n                | ID idnest indiceidnest : DOT ID\n              | DOT ID idnestindice : LBRACKET arithExpr RBRACKETarraySize : LBRACKET INTLIT RBRACKET\n                 | LBRACKET RBRACKETtype : TYPE\n            | VOIDreturnType : type\n                  | VOIDfParams : ID COLON type arraySize COMMA fParams\n               |aParams : expr COMMA aParams\n               |assignOp : EQrelOp : EQ\n             | NEQ\n             | LT\n             | GT\n             | LEaddOp : ADD\n             | SUBmultOp : MULT\n              | DIV'
    
_lr_action_items = {'CLASS':([0,],[5,]),'FUNCTION':([0,12,32,33,34,52,85,86,87,119,135,137,153,159,163,],[7,35,35,-6,-7,35,-61,-60,-59,-12,-10,-13,-11,-9,-14,]),'$end':([1,2,3,4,9,13,31,38,53,],[0,-1,-2,-3,-15,-21,-4,-22,-5,]),'ID':([5,10,11,35,37,39,40,41,42,43,44,45,46,48,66,67,68,80,89,91,92,93,94,95,96,97,98,99,100,101,102,112,116,117,128,160,],[8,28,29,54,56,69,-67,69,69,74,69,69,69,81,69,69,69,69,120,69,-68,-69,-70,-71,-72,69,-73,-74,69,-75,-76,69,120,120,28,120,]),'LBRACE':([6,8,84,85,86,87,107,115,126,148,150,158,],[10,12,-18,-61,-60,-59,128,-16,128,128,-19,-17,]),'LPAREN':([7,23,24,25,26,27,28,39,40,41,42,44,45,46,55,66,67,68,69,80,83,88,91,92,93,94,95,96,97,98,99,100,101,102,112,],[11,41,42,43,44,45,46,66,-67,66,66,66,66,66,89,66,66,66,46,66,116,117,66,-68,-69,-70,-71,-72,66,-73,-74,66,-75,-76,66,]),'RBRACE':([10,12,14,15,16,17,18,19,20,21,32,33,34,47,52,57,58,59,60,61,62,63,64,65,79,81,85,86,87,104,105,108,109,110,111,114,119,122,123,124,125,127,130,135,137,140,149,153,157,159,163,],[13,31,38,-23,-24,-25,-26,-27,-28,-29,53,-6,-7,-52,-8,-45,-30,-38,-39,-42,-44,-46,-47,-48,-53,-54,-61,-60,-59,-50,-51,-33,-34,-35,-36,-55,-12,-40,-41,-43,-49,-32,-56,-10,-13,149,-37,-11,-31,-9,-14,]),'IF':([10,128,],[23,23,]),'WHILE':([10,128,],[24,24,]),'READ':([10,128,],[25,25,]),'WRITE':([10,128,],[26,26,]),'RETURN':([10,128,],[27,27,]),'RPAREN':([11,29,46,47,50,57,59,60,61,62,63,64,65,70,72,73,75,76,77,79,81,89,103,104,105,111,112,114,116,117,118,122,123,124,125,129,130,132,133,160,164,],[30,49,-66,-52,83,-45,-38,-39,-42,-44,-46,-47,-48,106,107,108,109,110,111,-53,-54,119,125,-50,-51,-36,-66,-55,131,134,135,-40,-41,-43,-49,-65,-56,142,143,-64,-63,]),'CONSTRUCTOR':([12,32,33,34,52,85,86,87,119,135,137,153,159,163,],[36,36,-6,-7,36,-61,-60,-59,-12,-10,-13,-11,-9,-14,]),'TYPE':([12,32,33,34,51,52,82,85,86,87,90,119,135,136,137,141,144,151,152,153,159,163,],[37,37,-6,-7,87,37,87,-61,-60,-59,121,-12,-10,87,-13,87,87,87,87,-11,-9,-14,]),'EQ':([22,47,57,59,61,62,63,64,65,71,79,81,104,105,111,114,123,124,125,130,],[40,-52,-45,92,-42,-44,-46,-47,-48,92,-53,-54,-50,-51,-36,-55,-41,-43,-49,-56,]),'DOT':([28,69,74,81,],[48,48,48,48,]),'ARROW':([30,49,131,134,142,143,],[51,82,141,144,151,152,]),'COLON':([36,54,56,120,],[55,88,90,136,]),'INTLIT':([39,40,41,42,44,45,46,66,67,68,80,91,92,93,94,95,96,97,98,99,100,101,102,112,138,155,],[64,-67,64,64,64,64,64,64,64,64,64,64,-68,-69,-70,-71,-72,64,-73,-74,64,-75,-76,64,147,161,]),'FLOATLIT':([39,40,41,42,44,45,46,66,67,68,80,91,92,93,94,95,96,97,98,99,100,101,102,112,],[65,-67,65,65,65,65,65,65,65,65,65,65,-68,-69,-70,-71,-72,65,-73,-74,65,-75,-76,65,]),'NOT':([39,40,41,42,44,45,46,66,67,68,80,91,92,93,94,95,96,97,98,99,100,101,102,112,],[67,-67,67,67,67,67,67,67,67,67,67,67,-68,-69,-70,-71,-72,67,-73,-74,67,-75,-76,67,]),'SIGN':([39,40,41,42,44,45,46,66,67,68,80,91,92,93,94,95,96,97,98,99,100,101,102,112,],[68,-67,68,68,68,68,68,68,68,68,68,68,-68,-69,-70,-71,-72,68,-73,-74,68,-75,-76,68,]),'MULT':([47,57,62,63,64,65,79,81,104,105,111,114,125,130,],[-52,-45,101,-46,-47,-48,-53,-54,-50,-51,-36,-55,-49,-56,]),'DIV':([47,57,62,63,64,65,79,81,104,105,111,114,125,130,],[-52,-45,102,-46,-47,-48,-53,-54,-50,-51,-36,-55,-49,-56,]),'ADD':([47,57,61,62,63,64,65,79,81,104,105,111,114,124,125,130,],[-52,-45,98,-44,-46,-47,-48,-53,-54,-50,-51,-36,-55,-43,-49,-56,]),'SUB':([47,57,61,62,63,64,65,79,81,104,105,111,114,124,125,130,],[-52,-45,99,-44,-46,-47,-48,-53,-54,-50,-51,-36,-55,-43,-49,-56,]),'NEQ':([47,57,59,61,62,63,64,65,71,79,81,104,105,111,114,123,124,125,130,],[-52,-45,93,-42,-44,-46,-47,-48,93,-53,-54,-50,-51,-36,-55,-41,-43,-49,-56,]),'LT':([47,57,59,61,62,63,64,65,71,79,81,104,105,111,114,123,124,125,130,],[-52,-45,94,-42,-44,-46,-47,-48,94,-53,-54,-50,-51,-36,-55,-41,-43,-49,-56,]),'GT':([47,57,59,61,62,63,64,65,71,79,81,104,105,111,114,123,124,125,130,],[-52,-45,95,-42,-44,-46,-47,-48,95,-53,-54,-50,-51,-36,-55,-41,-43,-49,-56,]),'LE':([47,57,59,61,62,63,64,65,71,79,81,104,105,111,114,123,124,125,130,],[-52,-45,96,-42,-44,-46,-47,-48,96,-53,-54,-50,-51,-36,-55,-41,-43,-49,-56,]),'COMMA':([47,57,59,60,61,62,63,64,65,78,79,81,104,105,111,114,122,123,124,125,130,154,162,165,],[-52,-45,-38,-39,-42,-44,-46,-47,-48,112,-53,-54,-50,-51,-36,-55,-40,-41,-43,-49,-56,160,-58,-57,]),'RBRACKET':([47,57,61,62,63,64,65,79,81,104,105,111,113,114,123,124,125,130,147,155,161,],[-52,-45,-42,-44,-46,-47,-48,-53,-54,-50,-51,-36,130,-55,-41,-43,-49,-56,156,162,165,]),'LBRACKET':([47,81,87,114,121,145,146,],[80,-54,-59,-55,138,155,-60,]),'VOID':([51,82,136,141,144,151,152,],[86,86,146,86,86,86,86,]),'THEN':([106,],[126,]),'SEMICOLON':([121,156,],[137,163,]),'ELSE':([139,149,],[148,-37,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'prog':([0,],[1,]),'buildClassOrFunc':([0,],[2,]),'classDecl':([0,],[3,]),'funcDef':([0,],[4,]),'funcHead':([0,],[6,]),'funcBody':([6,],[9,]),'statement':([10,128,],[14,140,]),'assignStat':([10,128,],[15,15,]),'ifStat':([10,128,],[16,16,]),'whileStat':([10,128,],[17,17,]),'readStat':([10,128,],[18,18,]),'writeStat':([10,128,],[19,19,]),'returnStat':([10,128,],[20,20,]),'functionCall':([10,39,41,42,44,45,46,66,67,68,80,91,97,100,112,128,],[21,63,63,63,63,63,63,63,63,63,63,63,63,63,63,21,]),'variable':([10,39,41,42,43,44,45,46,66,67,68,80,91,97,100,112,128,],[22,57,57,57,73,57,57,57,57,57,57,57,57,57,57,57,22,]),'memberDecl':([12,32,52,],[32,52,52,]),'memberFuncDecl':([12,32,52,],[33,33,33,]),'memberVarDecl':([12,32,52,],[34,34,34,]),'assignOp':([22,],[39,]),'idnest':([28,69,74,81,],[47,47,47,114,]),'construct':([29,],[50,]),'expr':([39,44,45,46,112,],[58,75,76,78,78,]),'arithExpr':([39,41,42,44,45,46,66,80,91,97,112,],[59,71,71,59,59,59,103,113,122,123,59,]),'relExpr':([39,41,42,44,45,46,112,],[60,70,72,60,60,60,60,]),'term':([39,41,42,44,45,46,66,80,91,97,100,112,],[61,61,61,61,61,61,61,61,61,61,124,61,]),'factor':([39,41,42,44,45,46,66,67,68,80,91,97,100,112,],[62,62,62,62,62,62,62,104,105,62,62,62,62,62,]),'aParams':([46,112,],[77,129,]),'indice':([47,],[79,]),'returnType':([51,82,141,144,151,152,],[84,115,150,153,158,159,]),'type':([51,82,136,141,144,151,152,],[85,85,145,85,85,85,85,]),'relOp':([59,71,],[91,91,]),'addOp':([61,],[97,]),'multOp':([62,],[100,]),'fParams':([89,116,117,160,],[118,132,133,164,]),'statBlock':([107,126,148,],[127,139,157,]),'arraySize':([145,],[154,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> prog","S'",1,None,None,None),
  ('prog -> buildClassOrFunc','prog',1,'p_prog','hello.py',79),
  ('buildClassOrFunc -> classDecl','buildClassOrFunc',1,'p_buildClassOrFunc','hello.py',84),
  ('buildClassOrFunc -> funcDef','buildClassOrFunc',1,'p_buildClassOrFunc','hello.py',85),
  ('classDecl -> CLASS ID LBRACE RBRACE','classDecl',4,'p_classDecl','hello.py',89),
  ('classDecl -> CLASS ID LBRACE memberDecl RBRACE','classDecl',5,'p_classDecl','hello.py',90),
  ('memberDecl -> memberFuncDecl','memberDecl',1,'p_memberDecl','hello.py',94),
  ('memberDecl -> memberVarDecl','memberDecl',1,'p_memberDecl','hello.py',95),
  ('memberDecl -> memberDecl memberDecl','memberDecl',2,'p_memberDecl','hello.py',96),
  ('memberFuncDecl -> FUNCTION ID COLON LPAREN fParams RPAREN ARROW returnType','memberFuncDecl',8,'p_memberFuncDecl','hello.py',100),
  ('memberFuncDecl -> CONSTRUCTOR COLON LPAREN fParams RPAREN','memberFuncDecl',5,'p_memberFuncDecl','hello.py',101),
  ('memberFuncDecl -> FUNCTION ID COLON LPAREN RPAREN ARROW returnType','memberFuncDecl',7,'p_memberFuncDecl','hello.py',102),
  ('memberFuncDecl -> CONSTRUCTOR COLON LPAREN RPAREN','memberFuncDecl',4,'p_memberFuncDecl','hello.py',103),
  ('memberVarDecl -> TYPE ID COLON TYPE SEMICOLON','memberVarDecl',5,'p_memberVarDecl','hello.py',107),
  ('memberVarDecl -> TYPE ID COLON TYPE LBRACKET INTLIT RBRACKET SEMICOLON','memberVarDecl',8,'p_memberVarDecl','hello.py',108),
  ('funcDef -> funcHead funcBody','funcDef',2,'p_funcDef','hello.py',112),
  ('funcHead -> FUNCTION LPAREN ID RPAREN ARROW returnType','funcHead',6,'p_funcHead','hello.py',116),
  ('funcHead -> FUNCTION LPAREN ID construct RPAREN LPAREN fParams RPAREN ARROW returnType','funcHead',10,'p_funcHead','hello.py',117),
  ('funcHead -> FUNCTION LPAREN RPAREN ARROW returnType','funcHead',5,'p_funcHead','hello.py',118),
  ('funcHead -> FUNCTION LPAREN ID construct RPAREN LPAREN RPAREN ARROW returnType','funcHead',9,'p_funcHead','hello.py',119),
  ('construct -> <empty>','construct',0,'p_construct','hello.py',123),
  ('funcBody -> LBRACE RBRACE','funcBody',2,'p_funcBody','hello.py',127),
  ('funcBody -> LBRACE statement RBRACE','funcBody',3,'p_funcBody','hello.py',128),
  ('statement -> assignStat','statement',1,'p_statement','hello.py',132),
  ('statement -> ifStat','statement',1,'p_statement','hello.py',133),
  ('statement -> whileStat','statement',1,'p_statement','hello.py',134),
  ('statement -> readStat','statement',1,'p_statement','hello.py',135),
  ('statement -> writeStat','statement',1,'p_statement','hello.py',136),
  ('statement -> returnStat','statement',1,'p_statement','hello.py',137),
  ('statement -> functionCall','statement',1,'p_statement','hello.py',138),
  ('assignStat -> variable assignOp expr','assignStat',3,'p_assignStat','hello.py',142),
  ('ifStat -> IF LPAREN relExpr RPAREN THEN statBlock ELSE statBlock','ifStat',8,'p_ifStat','hello.py',147),
  ('whileStat -> WHILE LPAREN relExpr RPAREN statBlock','whileStat',5,'p_whileStat','hello.py',152),
  ('readStat -> READ LPAREN variable RPAREN','readStat',4,'p_readStat','hello.py',157),
  ('writeStat -> WRITE LPAREN expr RPAREN','writeStat',4,'p_writeStat','hello.py',162),
  ('returnStat -> RETURN LPAREN expr RPAREN','returnStat',4,'p_returnStat','hello.py',167),
  ('functionCall -> ID LPAREN aParams RPAREN','functionCall',4,'p_functionCall','hello.py',172),
  ('statBlock -> LBRACE statement RBRACE','statBlock',3,'p_statBlock','hello.py',177),
  ('expr -> arithExpr','expr',1,'p_expr','hello.py',182),
  ('expr -> relExpr','expr',1,'p_expr','hello.py',183),
  ('relExpr -> arithExpr relOp arithExpr','relExpr',3,'p_relExpr','hello.py',188),
  ('arithExpr -> term addOp arithExpr','arithExpr',3,'p_arithExpr','hello.py',193),
  ('arithExpr -> term','arithExpr',1,'p_arithExpr','hello.py',194),
  ('term -> factor multOp term','term',3,'p_term','hello.py',199),
  ('term -> factor','term',1,'p_term','hello.py',200),
  ('factor -> variable','factor',1,'p_factor','hello.py',205),
  ('factor -> functionCall','factor',1,'p_factor','hello.py',206),
  ('factor -> INTLIT','factor',1,'p_factor','hello.py',207),
  ('factor -> FLOATLIT','factor',1,'p_factor','hello.py',208),
  ('factor -> LPAREN arithExpr RPAREN','factor',3,'p_factor','hello.py',209),
  ('factor -> NOT factor','factor',2,'p_factor','hello.py',210),
  ('factor -> SIGN factor','factor',2,'p_factor','hello.py',211),
  ('variable -> ID idnest','variable',2,'p_variable','hello.py',216),
  ('variable -> ID idnest indice','variable',3,'p_variable','hello.py',217),
  ('idnest -> DOT ID','idnest',2,'p_idnest','hello.py',222),
  ('idnest -> DOT ID idnest','idnest',3,'p_idnest','hello.py',223),
  ('indice -> LBRACKET arithExpr RBRACKET','indice',3,'p_indice','hello.py',228),
  ('arraySize -> LBRACKET INTLIT RBRACKET','arraySize',3,'p_arraySize','hello.py',233),
  ('arraySize -> LBRACKET RBRACKET','arraySize',2,'p_arraySize','hello.py',234),
  ('type -> TYPE','type',1,'p_type','hello.py',239),
  ('type -> VOID','type',1,'p_type','hello.py',240),
  ('returnType -> type','returnType',1,'p_returnType','hello.py',245),
  ('returnType -> VOID','returnType',1,'p_returnType','hello.py',246),
  ('fParams -> ID COLON type arraySize COMMA fParams','fParams',6,'p_fParams','hello.py',251),
  ('fParams -> <empty>','fParams',0,'p_fParams','hello.py',252),
  ('aParams -> expr COMMA aParams','aParams',3,'p_aParams','hello.py',256),
  ('aParams -> <empty>','aParams',0,'p_aParams','hello.py',257),
  ('assignOp -> EQ','assignOp',1,'p_assignOp','hello.py',262),
  ('relOp -> EQ','relOp',1,'p_relOp','hello.py',267),
  ('relOp -> NEQ','relOp',1,'p_relOp','hello.py',268),
  ('relOp -> LT','relOp',1,'p_relOp','hello.py',269),
  ('relOp -> GT','relOp',1,'p_relOp','hello.py',270),
  ('relOp -> LE','relOp',1,'p_relOp','hello.py',271),
  ('addOp -> ADD','addOp',1,'p_addOp','hello.py',276),
  ('addOp -> SUB','addOp',1,'p_addOp','hello.py',277),
  ('multOp -> MULT','multOp',1,'p_multOp','hello.py',282),
  ('multOp -> DIV','multOp',1,'p_multOp','hello.py',283),
]
