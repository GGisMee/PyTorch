# Open the input file in read mode
with open("more/input.txt", "r") as input_file:
    # Read all lines from the input file
    lines = input_file.readlines()

# Open the input file in write mode to overwrite its content
with open("more/input.txt", "w") as output_file:
    # Iterate over each line
    for i, line in enumerate(lines, start=1):
        # Strip any trailing whitespace from the line
        line = line.rstrip()
        # Append the word "apple" followed by the current line number
        modified_line = f"{line} apple{i}\n"
        # Write the modified line back to the file
        output_file.write(modified_line)
