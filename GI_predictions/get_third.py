def extract_third_lines(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Input file not found.")
        return

    third_lines = [line.strip() for i, line in enumerate(lines) if (i + 1) % 3 == 0]

    if not third_lines:
        print("No third lines found in the input file.")
        return

    try:
        with open(output_file, 'w') as f:
            for line in third_lines:
                f.write(line + '\n')
        print(f"New file '{output_file}' created with extracted third lines.")
    except IOError:
        print("Error occurred while writing to the output file.")


if __name__ == "__main__":
    input_file = input("Enter the path to the input file: ")
    output_file = input("Enter the path for the output file: ")+".txt"

    extract_third_lines(input_file, output_file)
