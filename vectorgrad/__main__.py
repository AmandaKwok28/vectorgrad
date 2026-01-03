import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="vectorgrad CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("test", help="Run test suite")
    subparsers.add_parser("lint", help="Run lint checks")
    
    args = parser.parse_args()
    
    if args.command == "test":
        subprocess.run([sys.executable, "-m", "pytest"], check=True)
    else:
        parser.print_help()
        

if __name__ == "__main__":
    main()