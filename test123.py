import os

# Get the current directory path
current_dir = os.getcwd()

# List all files
files = os.listdir(current_dir)

# Print them out to find the culprit
print("Files in directory:")
for f in files:
    # Check for reserved names that might be causing the issue
    if f.upper() in ['PRN', 'CON', 'AUX', 'NUL', 'COM1', 'LPT1']:
        print(f"!!! FOUND CULPRIT: {f} !!!")
    else:
        print(f)