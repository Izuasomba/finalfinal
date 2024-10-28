import os
import sys
import subprocess

# Check if a script name was provided
if len(sys.argv) < 2:
    print("Usage: python run.py <script_name.py>")
    sys.exit(1)

script_name = sys.argv[1]

# Construct the path to the script
script_path = os.path.join('scripts', script_name)

# Check if the script exists
if os.path.isfile(script_path):
    # Use the same Python interpreter that is running this script
    python_executable = sys.executable
    
    # Run the script
    subprocess.run([python_executable, script_path] + sys.argv[2:])
else:
    print(f"Script '{script_name}' not found in the 'scripts' folder.")
