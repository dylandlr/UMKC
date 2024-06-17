import os

import ollama






def get_completion(prompt):

    messages = [{

    'role' : 'user',

    'content' : prompt

    }]

    response = ollama.chat(model='llama3', messages = messages)

    return response['message']['content']

    




def process_file(input_path, out_directory):

    file_name = os.path.splitext(os.path.basename(input_path))[0]

    try:

        with open(input_path, 'r') as file:

            content = file.read()

        prompt = f"""

        {content}

        #Context#

        The Rust code is suspected to have a vulnerability, but no specific project details or past issues are known. This absence of context necessitates a comprehensive and focused review to identify the most significant security concern, covering a wide
        range of common vulnerabilities in Rust or CVE description.

        #Objective#

        The objective is to perform a detailed scan of the entire code to pinpoint the section most likely to contain a critical vulnerability.

        #Style#

        Request a detailed analysis in a technical report format. The report should focus exclusively on the most significant vulnerable code snippet, annotated with comments that explain why this section is particularly problematic.

        #Tone#

        Adopt a professional and investigative tone, reflecting the systematic approach required to uncover and document the most critical vulnerability.

        #Audience#

        The intended audience is people who have a good understanding of Rust and are familiar with common vulnerabilities. The report should provide insights that are technically thorough yet accessible to these professionals.

        #Response#

        The response should be a structured report that focuses solely on the most critical vulnerability identified, with a word limit of 500 words.  It should only provide a clear description of the vulnerability and its location in the code. The report should
        not include any recommendations for mitigation, impact, conclusion, or suggestion.

        """




        response = get_completion(prompt)

        if response is None:

            print("Not OK 1: " + file_name)

            return

        output_file_path = os.path.join(out_directory, file_name + ".txt")

        with open(output_file_path, 'w') as output_file:

            output_file.write(response)

        print("OK: " + file_name)

    except Exception as e:

        print(f"Error processing file {input_path}: {e}")

        print("Not OK 2: " + file_name)




def process_files_in_directory(directory_path, out_directory):

    for root, _, files in os.walk(directory_path):

        relative_path = os.path.relpath(root, directory_path)

        current_out_directory = os.path.join(out_directory, relative_path)

        if not os.path.exists(current_out_directory):

            os.makedirs(current_out_directory)




        for file_name in files:
# use the correct file abbreviation:
            if file_name.endswith(".txt"): 

                input_file_path = os.path.join(root, file_name)

                process_file(input_file_path, current_out_directory)






if __name__ == "__main__":

    root_directory = "../data/code/c"  # Set your source directory path here

    out_directory = "../results"  # Set your output directory path here
    
    process_files_in_directory(root_directory, out_directory)