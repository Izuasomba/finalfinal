# scripts/__main__.py
import os
import sys


# Get the script name from the command line arguments
if len(sys.argv) > 1:
    script_name = sys.argv[1]
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if os.path.isfile(script_path):
        os.execv(sys.executable, [sys.executable, script_path] + sys.argv[2:])
    else:
        print(f"Script '{script_name}' not found in scripts folder.")
else:
    print("Please provide a script name to run.")
