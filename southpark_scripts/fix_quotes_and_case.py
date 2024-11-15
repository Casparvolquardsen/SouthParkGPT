from southpark_scripts.clean_dataset_with_gpt4 import find_csv_files


def fix_quotes(line: str) -> str:
    if line.startswith(','):
        return line
    else:
        current_punctuation = None
        current_start_index = -1
        open_quotes_after_bracket = False
        after_comma = False
        i = 0
        while i < len(line):
            if not after_comma and line[i] == ',':
                after_comma = True
                i += 1
                continue
            elif not after_comma:
                i += 1
                continue
            else:
                if line[i] in ['"', "[", "]"]:
                    if current_punctuation is None:
                        if line[i] in ['"', "["]:
                            current_punctuation = line[i]
                            current_start_index = i
                    elif current_punctuation == '"':
                        if line[i] == '"':
                            current_punctuation = None
                            current_start_index = -1
                        elif line[i] == '[':
                            # close the quotes and make sure that a space is added after the quotes and to open the
                            # quotes again after the brackets

                            # check if the quotes are started directly before the bracket
                            if i - 1 == current_start_index:
                                # remove the starting quote
                                line = line[: i - 1] + line[i:]
                                open_quotes_after_bracket = True
                                current_punctuation = '['
                                current_start_index = i - 1
                                continue
                            else:
                                if line[i - 1] != ' ':
                                    line = line[:i] + ' ' + line[i:]
                                    i += 1
                                line = line[: i - 1] + '"' + line[i - 1 :]
                                i += 1
                                current_punctuation = '['
                                current_start_index = i
                                open_quotes_after_bracket = True

                    elif current_punctuation == '[':
                        if line[i] == ']':
                            current_punctuation = None
                            current_start_index = -1
                            if open_quotes_after_bracket:
                                open_quotes_after_bracket = False

                                # only add a quote if it is not the end of the line
                                if i < len(line) - 3:
                                    if line[i + 1] != ' ':
                                        line = line[: i + 1] + ' "' + line[i + 1 :]
                                    else:
                                        line = line[: i + 2] + '"' + line[i + 2 :]

                i += 1
        if current_punctuation == '"':
            line = line[:current_start_index] + line[current_start_index + 1 :]
        return line


def remove_intermediate_quotes(line: str) -> str:
    if line.startswith(','):
        return line
    else:
        i = 0
        # remove all quotes
        while i < len(line):
            if line[i] == '"':
                line = line[:i] + line[i + 1 :]
                continue
            i += 1

        # add quotes at the beginning and end of speech
        i = 0
        while i < len(line):
            if line[i] == ',':
                break
            i += 1

        if i > len(line) - 2:
            print(line)
            return line

        if line[i + 1] != ' ':  # add space after comma
            line = line[: i + 1] + ' ' + line[i + 1 :]

        # beginning
        line = line[: i + 2] + '"' + line[i + 2 :]

        # end
        line = line[: len(line) - 1] + '"' + line[len(line) - 1 :]

        return line


def fix_case(line: str) -> str:
    for i in range(len(line)):
        if line[i] == '[':
            # make beginning in brackets uppercase
            beginn_char = line[i + 1]
            if beginn_char.islower():
                line = line[: i + 1] + beginn_char.upper() + line[i + 2 :]

    return line


def fix_brackets(line: str) -> str:
    if ',' not in line and line.startswith('['):
        # remove brackets in the line
        i = 0
        while i < len(line):
            if line[i] in ['[', ']']:
                line = line[:i] + line[i + 1 :]
            else:
                i += 1
        # add , at the beginning of the line
        line = ',' + line

    return line


def fix_invalid_characters(line):
    line = line.replace('’', "'")
    line = line.replace('‘', "'")
    line = line.replace('`', "'")
    line = line.replace('…', '...')
    line = line.replace('—', '-')
    line = line.replace('–', '-')
    line = line.replace('”', '"')
    line = line.replace('“', '"')
    line = line.replace('é', 'e')

    return line


if __name__ == "__main__":
    # Path to the input script file
    file_paths = find_csv_files(
        '/Users/caspar/Repositories/ml-project-24/southpark_scripts/all_scripts_cleaned'
    )

    allowed_chars = list(
        '\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}'
    )

    invalid_parts = dict()

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            original_script = file.readlines()

        if not original_script:
            continue

        if original_script[0] == '\n':
            original_script.pop(0)

        invalid_chars = set()
        invalid_lines = set()

        for i in range(len(original_script)):
            fixed_line = fix_quotes(original_script[i])
            # fixed_line = remove_intermediate_quotes(original_script[i])
            fixed_line = fix_case(fixed_line)
            fixed_line = fix_brackets(fixed_line)
            fixed_line = fix_invalid_characters(fixed_line)
            original_script[i] = fixed_line

            for char in fixed_line:
                if char not in allowed_chars:
                    invalid_chars.add(char)
                    invalid_lines.add(i)

        if invalid_chars:
            print(file_path + ' has invalid characters: ' + ' '.join(invalid_chars))

        if invalid_lines:
            invalid_parts[file_path] = list(invalid_lines)

        # remove empty lines of end lines
        i = 0
        while i < len(original_script):
            if original_script[i] == '\n' or (
                i == len(original_script) - 1 and original_script[i].startswith(',End ')
            ):
                original_script.pop(i)
            else:
                i += 1

        with open(file_path, 'w') as file:
            file.writelines(original_script)

    print(invalid_parts)
