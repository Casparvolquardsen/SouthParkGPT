import glob
import os

from openai import OpenAI
from tqdm import tqdm


def find_csv_files(directory):
    # Use glob to find all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    return csv_files


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def write_file(file_path, file_text):
    with open(file_path, 'w') as file:
        file.write(file_text)


# Function to correct the script using OpenAI API
def correct_script(file_path, client):
    # Read the original script from the file
    original_script = read_file(file_path)

    # script parts, split the script every 100 lines
    original_script_lines = original_script.split('\n')
    original_script_parts = []

    num_lines = 0
    current_part = ''
    for line in original_script_lines:
        current_part += line + '\n'
        num_lines += 1
        if num_lines == 70:
            original_script_parts.append(current_part)
            current_part = ''
            num_lines = 0

    if current_part:
        original_script_parts.append(current_part)

    # check that the last part is not too short
    if len(original_script_parts[-1].split('\n')) < 40:
        original_script_parts[-2] += original_script_parts[-1]
        original_script_parts.pop(-1)

    corrected_script = ''

    # Correct the script part by part
    for part in original_script_parts:
        # Define the prompt
        prompt = read_file('cleaning_instruction_prompt.txt')

        # Append the original script to the prompt
        prompt += part

        # Set up the OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )

        if response.choices[0].finish_reason == 'stop':
            corrected_script += '\n' + response.choices[0].message.content
        else:
            corrected_script += '\n' + part

    return corrected_script


# Main function to run the script
if __name__ == "__main__":
    # You need to set up the OpenAI API key as an environment variable OPENAI_API_KEY
    client = OpenAI()

    # Path to the input script file
    file_paths = find_csv_files('all_scripts_cleaned/')

    for file_path in tqdm(file_paths):
        # Get the corrected script
        corrected_script = correct_script(file_path, client)

        # Clean up the corrected script
        corrected_script = corrected_script.replace('\n\n', '\n')
        corrected_script = corrected_script.replace('\n, ', '\n,')

        # overwrite the original file with the corrected script
        write_file(file_path, corrected_script)
