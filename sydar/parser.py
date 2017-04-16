"""
.. module:: parser
   :platform: Unix
   :synopsis: 

.. moduleauthor:: Mohammed Alshiekh


"""

from pyparsing import *
from symbol_table import SymbolTable
import sys
from region import *
from Automata import RegularLanguage

def printf(text):
   print text

def tup2trans(string):
    tup = eval(string)
    new_tup = ('q_{}'.format(tup[0]),'q_{}'.format(tup[1]))
    return new_tup

def parse_miu(file_):
    parser = Parser()
    aut_begin = 0
    constants_begin = 0
    system_begin = 0
    specs_begin = 0
    ap_begin = 0
    let_begin = 0
    # getting blocks line numbers
    f = open(file_, 'r').readlines()
    for i,line in enumerate(f):
        # if 'dimensions' in line[:len('dimensions')] and not 'end' in line :
        #     dim_begin = i+1
        # if 'dimensions' in line and 'end' in line[:len('end')]:
        #     dim_end = i
        if 'constants' in line[:len('constants')] and not 'end' in line:
            constants_begin = i+1
        if 'constants' in line and 'end' in line[:len('end')]:
            constants_end = i
        if 'regions' in line[:len('regions')] and not 'end' in line:
            region_begin = i+1
        if 'regions' in line and 'end' in line[:len('end')]:
            region_end = i
        if 'system' in line[:len('system')] and not 'end' in line:
            system_begin = i+1
        if 'system' in line and 'end' in line[:len('end')]:
            system_end = i
        if 'automaton' in line[:len('automaton')] and not 'end' in line:
            aut_begin = i+1
        if 'automaton' in line and 'end' in line[:len('end')]:
            aut_end = i
        if 'specifications' in line[:len('specifications')] and not 'end' in line:
            specs_begin = i+1
        if 'specifications' in line and 'end' in line[:len('end')]:
            specs_end = i
        if 'ap' in line[:len('ap')] and not 'end' in line:
            ap_begin = i+1
        if 'ap' in line and 'end' in line[:len('end')]:
            ap_end = i
        if 'letters' in line[:len('letters')] and not 'end' in line:
            let_begin = i+1
        if 'letters' in line and 'end' in line[:len('end')]:
            let_end = i
    
    if constants_begin > 0:
        parser.const_rule.parseString(''.join(f[constants_begin:constants_end]))    
    
    if system_begin > 0:
        parser.system_rule.parseString(''.join(f[system_begin:system_end]))
    
    if region_begin > 0:
        parser.region_rule.parseString(''.join(f[region_begin:region_end]))
    
    if ap_begin > 0:
        parser.ap_rule.parseString(''.join(f[ap_begin:ap_end]))
    
    if let_begin > 0:
        parser.let_rule.parseString(''.join(f[let_begin:let_end]))
    
    if specs_begin > 0:
        parser.spec_rule.parseString(''.join(f[specs_begin:specs_end]))
    
    if aut_begin > 0:
        parser.aut_rule.parseString(''.join(f[aut_begin:aut_end]))
        return parser.symbol_table
    else:
        return parser.symbol_table.rename_states() 

class Parser(object):
    """
    This class creates an instance that is used to parse a \*.miu file.

    Args:
        None

    Attributes:
        | symbol_table (SymbolTable):
        | const_rule: The rules for parsing the Constant block.
        | let_rule: The rules for parsing the Letter block.
        | spec_rule: The rules for parsing the Spec block.
        | region_rule: The rules for parsing the Region block.
        | ap_rule: The rules for parsing the AP block.
        | system_rule: The rules for parsing the System block.
    """
    def __init__(self):

        self.symbol_table = SymbolTable()     
        comment = (Literal('#') + restOfLine).suppress()
        point = Literal( "." )
        num = Combine(Optional("-") + Word( nums, nums ) + 
                           Optional( point + Optional( Word( nums ) ) ))
        semi = Literal(';').suppress()
        equals = Literal('=').suppress()
        bind = Literal('<-').suppress()
        negate = Literal('-')

        variable_lhs = Word(alphas, alphanums + '_')
        variable_rhs = Word(alphas, alphanums + '_').setParseAction(lambda s, loc, toks: self.symbol_table.lookup(toks[0],'constant'))
        negated_variable_rhs = (negate+Word(alphas, alphanums + '_')).setParseAction(lambda s, loc, toks: self.symbol_table.lookup(toks[1],'constant',neg=True))
        expr = Forward()
        matrix = Literal('[') + expr + ZeroOrMore(',' + expr) + Literal(']')
        listoflists = Literal('[') + matrix + ZeroOrMore(',' + matrix) + Literal(']') #Optional('[') + matrix + ZeroOrMore(',' + matrix) + Optional(']')
        operand = num| variable_rhs | matrix | negated_variable_rhs
        factor = operand | expr
        term = factor + ZeroOrMore( oneOf('* /') + factor )
        expr << Optional('(') + term + ZeroOrMore(oneOf('+ -') + term) + Optional(')') 

        # constant block assignment expression
        const_assigmentExp = (variable_lhs + equals + Combine(expr) + semi).setParseAction(lambda s, loc, toks:\
                                                                                           self.symbol_table.insert(toks[0],'constant',toks[1]))

        # Region block assignment expression
        region_expr = Forward()
        region_variable_rhs = Word(alphas, alphanums + '_').setParseAction(lambda s, loc, toks: self.symbol_table.lookup(toks[0],'region'))
        region_term = MatchFirst([(Optional('(') + ((Literal('HalfSpace(') | Literal('Ellipsoid(')) + Combine(expr)\
                       + Literal(',') + Combine(expr) + Optional(')')  + Optional(')')) | Literal('Empty()') | Literal('Workspace()') ) ,\
                                  (Optional('(') + region_variable_rhs + Optional(')'))])
        region_expr << Optional('(') + region_term + ZeroOrMore(oneOf('& |') + region_term) + Optional(')')
        region_assigmentExp = (variable_lhs + equals + Combine(region_expr) + semi).setParseAction(lambda s, loc, toks: \
                                                                                                   self.symbol_table.insert(toks[0],'region',toks[1]))

        # System block assignment expression
        system_assigmentExp = (variable_lhs + equals + Combine(expr) + semi).setParseAction(lambda s, loc, toks:\
                                                                                            self.symbol_table.insert(toks[0],'system',toks[1]))

        # AP block assignment expression
        ap_expr = Optional('(') + region_term + ZeroOrMore(oneOf('& |') + region_term) + Optional(')')
        ap_assigmentExp = (variable_lhs + bind + Combine(ap_expr) + semi).setParseAction(lambda s, loc, toks:\
                                                                                         self.symbol_table.insert(toks[0],'ap',toks[1]))

        # Spec block assignment expression
        spec_term = MatchFirst([Word(alphas,'*'),Word(alphas,exact=1)]).setParseAction(lambda s, loc, toks:self.check_letter(toks))
        spec_expr = (Optional('(') + Literal('RE(').setParseAction(lambda s, loc, toks:\
                      'RegularLanguage("') + ZeroOrMore(spec_term)+ Word(')').setParseAction(lambda s,\
                       loc, toks: '",self.container).recognizer.asDFA().to_graph(self.container)')  + Optional(')'))
        spec_assigmentExp = (variable_lhs + equals + Combine(spec_expr) + semi).setParseAction(lambda s, loc, toks:\
                                                                                               self.symbol_table.insert(toks[0],'spec',toks[1]))

        # Letter block assignment expression
        let_expr = MatchFirst([Combine(Word('{')+Word('}')).setParseAction(lambda s, loc, toks:"set()".format(toks[0])),Word('{') + Word(alphas, exact=1).setParseAction(lambda s, loc, toks:"'{}'".format(toks[0]))\
                   + ZeroOrMore(',' + Word(alphas).setParseAction(lambda s, loc, toks:"'{}'".format(toks[0]))) + Word('}')])

        let_assigmentExp = (variable_lhs + equals + Combine(let_expr) + semi).setParseAction(lambda s, loc, toks:\
                                                                                             self.symbol_table.insert(toks[0],'letter',toks[1]))

        # Automaton block assignment expression
        comma = Literal(',').suppress()
        left_bracket = Literal('[').suppress()
        right_bracket = Literal(']').suppress()
        region_rhs = Word(alphanums).setParseAction(lambda s, loc, toks: self.symbol_table.lookup(toks[0],'region'))
        #
        
        index = left_bracket+Word(nums)+right_bracket
        tup = Combine(Literal('(')+Word(nums)+Literal(',')+Word(nums)+Literal(')'))
        t_index = left_bracket+tup+right_bracket
        edges_list = left_bracket + ZeroOrMore(tup+comma) + tup + right_bracket
        alist = left_bracket + ZeroOrMore(Word(nums)+comma) + Word(nums) + right_bracket
        n_nodes_aut = Literal('n_nodes') + equals + Word(nums)
        edges_aut = Literal('edges') + equals + edges_list
        accepting_aut = Literal('accepting') + equals + alist
        initial_aut = Literal('initial') + equals + alist
        node_bind_aut = Literal('nodes')+ index + bind + region_rhs
        edge_bind_aut = Literal('edges')+ t_index + bind + region_rhs
        aut_variable_lhs = Literal('edges') |\
                           Literal('n_nodes') |\
                           Literal('accepting') |\
                           Literal('initial') |\
                           Literal('nodes[') + Literal(nums) + Literal(']')
        
        aut_assignmentExp = (n_nodes_aut.setParseAction(lambda s, loc, toks:\
                                                   self.symbol_table.insert(toks[0],'aut',toks[1])) |\
                            edges_aut.setParseAction(lambda s, loc, toks:\
                                                   self.symbol_table.insert(toks[0],'aut',toks[1:])) |\
                            accepting_aut.setParseAction(lambda s, loc, toks:\
                                                   self.symbol_table.insert(toks[0],'aut',toks[1:])) |\
                            initial_aut.setParseAction(lambda s, loc, toks:\
                                                   self.symbol_table.insert(toks[0],'aut',toks[1:])) |\
                            node_bind_aut.setParseAction(lambda s, loc, toks:\
                                                   self.symbol_table.insert('q_'+toks[1],'aut',toks[2])) |\
                            edge_bind_aut.setParseAction(lambda s, loc, toks:\
                                                   self.symbol_table.insert(tup2trans(toks[1]),'aut',toks[2])))\
                            + semi
                  
        const_parse = const_assigmentExp     
        system_parse = system_assigmentExp
        region_parse = region_assigmentExp
        ap_parse = ap_assigmentExp
        let_parse = let_assigmentExp
        spec_parse = spec_assigmentExp
        aut_parse = aut_assignmentExp

        self.const_rule = OneOrMore(Group(const_parse)) 
        self.const_rule.ignore(comment) 
        self.let_rule = OneOrMore(Group(let_parse))        
        self.let_rule.ignore(comment)
        self.spec_rule = OneOrMore(Group(spec_parse))         
        self.spec_rule.ignore(comment)
        self.ap_rule = OneOrMore(Group(ap_parse))        
        self.ap_rule.ignore(comment)
        self.region_rule = OneOrMore(Group(region_parse))        
        self.region_rule.ignore(comment)
        self.system_rule = OneOrMore(Group(system_parse))        
        self.system_rule.ignore(comment) 
        self.aut_rule = OneOrMore(Group(aut_parse))        
        self.aut_rule.ignore(comment) 
        
    def check_letter(self, letter):
        """
        This method will inform the user if the letter does not exist in symbol_table.container['letter'] keys, then quits.

        Args:
            letter (str): The letter to be checked if exist as a key.

        Returns:
            None
        """
        if letter[0][0] not in self.symbol_table.container['letter'].keys():
            print 'Letter {} is not defined'.format(letter[0][0])
            sys.exit('check_letter')

