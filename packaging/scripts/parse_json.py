import json
import os
import argparse
ABS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
json_file = os.path.join(ABS_DIR_PATH, 'build.property')

def check_combination(json_data, a, b):
    # Check if the given values (a, b) match any key-value pair in "args_combination"
    for key, values in json_data.get('args_combination', {}).items():
        if set([a, b]) == set(map(str, values)):
            return key
    return None

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Process JSON configuration file.')
    parser.add_argument('-p', '--python', help='python version', required=True)
    parser.add_argument('-v', '--pytorch', help='pytorch version', required=True)
    args = parser.parse_args()

    # load python and pytorch version
    python_version = args.python
    pytorch_version = args.pytorch

    # Load the JSON file
    with open(json_file) as f:
        data = json.load(f)
    
    if python_version == "all":
        python_version = [values[1] for values in data.get('args_combination', {}).values()]
    else:
        python_version = [python_version]
    if pytorch_version == "all":
        pytorch_version = [values[0] for values in data.get('args_combination', {}).values()]
    else:
        pytorch_version = [pytorch_version]
    
    for a, b in [(item1, item2) for item1 in pytorch_version for item2 in python_version]:
        result = check_combination(data, a, b)
        if result:
            print("matched ({}, {}).".format(a, b))
            # Create a filename based on the key
            filename = "pt{}-py{}.txt".format(a, b)
            
            # Save the values to the text file
            with open(filename, 'w') as txt_file:
                txt_file.write('{}\n{}'.format(a, b))
        else:
            print("No matching combination found for the values ({}, {}).".format(a, b))

if __name__ == "__main__":
    main()
