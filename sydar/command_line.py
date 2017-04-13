from parser import parse_miu
from cvx_gen import to_cvx
import argparse

def main():
    parser = argparse.ArgumentParser(description='Synthesis of Hybrid Systems.')
    parser.add_argument('input_file',type=str)
    parser.add_argument('-o','--output', help='Dumps the output to the specified file')
    args = parser.parse_args()
    symbol_table = parse_miu(args.input_file)    

    nodes = symbol_table.get_tagged_nodes()
    edges = symbol_table.get_tagged_edges()
    constants = symbol_table.container['system']
    output = to_cvx(nodes,edges,constants)

    if args.output is not None:
        with open(args.output, "w") as text_file:
            text_file.write(output)

    else:
        print output

