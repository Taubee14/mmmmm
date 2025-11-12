#!/Users/xuruoyao/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/789befcc0983c780ea7704e326f087b1/Message/MessageTemp/e47189fb02641419fdc65b06e7a6315f/File/mobile-master 2/.venv/bin/python3.12

import sys
import json
import argparse
from pprint import pformat

import jmespath
from jmespath import exceptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expression')
    parser.add_argument('-f', '--filename',
                        help=('The filename containing the input data.  '
                              'If a filename is not given then data is '
                              'read from stdin.'))
    parser.add_argument('--ast', action='store_true',
                        help=('Pretty print the AST, do not search the data.'))
    args = parser.parse_args()
    expression = args.expression
    if args.ast:
        # Only print the AST
        expression = jmespath.compile(args.expression)
        sys.stdout.write(pformat(expression.parsed))
        sys.stdout.write('\n')
        return 0
    if args.filename:
        with open(args.filename, 'r') as f:
            data = json.load(f)
    else:
        data = sys.stdin.read()
        data = json.loads(data)
    try:
        sys.stdout.write(json.dumps(
            jmespath.search(expression, data), indent=4, ensure_ascii=False))
        sys.stdout.write('\n')
    except exceptions.ArityError as e:
        sys.stderr.write("invalid-arity: %s\n" % e)
        return 1
    except exceptions.JMESPathTypeError as e:
        sys.stderr.write("invalid-type: %s\n" % e)
        return 1
    except exceptions.UnknownFunctionError as e:
        sys.stderr.write("unknown-function: %s\n" % e)
        return 1
    except exceptions.ParseError as e:
        sys.stderr.write("syntax-error: %s\n" % e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
