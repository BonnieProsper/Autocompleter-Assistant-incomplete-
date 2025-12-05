# test_cli.py - basic CLI smoke check
import sys
import os
import io
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cli import main

def run_cli(cmd):
    sys.stdin = io.StringIO(cmd + "\nexit\n")
    sys.argv = ["cli.py"]
    try:
        main()
    except SystemExit:
        pass

if __name__ == "__main__":
    run_cli("suggest python")
    run_cli("train data/sample.txt")
    print("cli test done.")
