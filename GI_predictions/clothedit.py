import numpy as np
with open('GI_predictions/clothingcolors.txt', 'r') as f:
    lines = f.readlines()
lines2 = []
for i in lines:
    lines2.append(str(i[:-1]))
new_lines = []
for i in lines2:
    line = [0,0,0,0,0,0,0,0]
    for i2 in i:
        #print(int(i2))
        line[int(i2)] = 1
    new_lines.append(line)




with open("GI_predictions/wa.txt", "r") as input_file:
    # Read all lines from the input file
    lines = input_file.readlines()

# Open the input file in write mode to overwrite its content
with open("GI_predictions/wa.txt", "w") as output_file:
    # Iterate over each line
    for i, line in enumerate(lines):

        # Strip any trailing whitespace from the line
        line = line.rstrip()
        # Append the corresponding line from new_lines

        modified_line = f"{line},{str(new_lines[i])[1:-1].replace(' ', '')}\n"
        print(modified_line)
        # Write the modified line back to the file
        output_file.write(modified_line)



# for i, el in enumerate(new_lines):
#     print(i, el)

#print(len(new_lines))
# with open('GI_predictions/wa.txt', 'r') as f:
#     lines = f.readlines()

#     for i, line in enumerate(lines, start = 1):
#         if i == 1:
#             print(line)
#             continue
#         i-=1
#         line = line.rstrip()
#         modified_line = f"{line},{str(new_lines[i])[1:-1].replace(' ', '')}".replace("\n", "")
#         print(modified_line)

#         pass
    # for line in new_lines:
    #     f.write()