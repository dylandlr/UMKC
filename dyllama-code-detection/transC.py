import os
import shutil

def process_files_in_directory(root_directory, out_directory):
    # Create the directories if they don't exist
    os.makedirs(os.path.join(out_directory, 'pos'), exist_ok=True)
    os.makedirs(os.path.join(out_directory, 'neg'), exist_ok=True)

    # Get the list of already processed files
    processed_files = set(os.listdir(os.path.join(out_directory, 'pos'))) | set(os.listdir(os.path.join(out_directory, 'neg')))

    # Walk through all files in the root directory
    for root, _, files in os.walk(root_directory):
        for file_name in files:
            # Use the correct file abbreviation
            if file_name.endswith(".txt"):
                input_file_path = os.path.join(root, file_name)
                # Determine the output directory based on the presence of 'pos' or 'neg' in the root
                current_out_directory = os.path.join(out_directory, 'pos' if 'pos' in root else 'neg')
                # Move the file if it hasn't been processed already
                if file_name not in processed_files:
                    shutil.move(input_file_path, current_out_directory)

if __name__ == "__main__":
    root_directory = "../data/umkc/c_functions"  # Set your source directory path here
    out_directory = "../dyllama-code-detection/out/c"  # Set your output directory path here
    process_files_in_directory(root_directory, out_directory)
