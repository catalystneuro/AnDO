import os
import os.path as op
import re
import argparse
import pathlib
from ando.rulesStructured import RULES_SET


def is_valid(input_directory):
    """Checks the validity of a data set with respect to the BIDS-animal-ephys specifications.

    The specifications that define what is checked by this function is available in the following document:
    https://docs.google.com/document/d/1oG-C8T-dWPqfVzL2W8HO3elWK8NIh2cOCPssRGv23n0


    Parameters
    ----------
        input_directory : string
            Name of the root directory containing the data set to be checked

    Returns
    -------
        valid : boolean
            True if the data set follows the ephys-BIDS specification; False if not

        error_list : list
            List of errors (empty if the data set is valid)

    """

    # remove ending / or \ if the input directory was given with it at the end
    input_directory = pathlib.Path(input_directory)
    # count the number of / or \ in the directory name to estimate its "depth"
    initial_depth = len(input_directory.resolve().parents)

    error_list = []

    # walk through the directory tree using the os.walk function
    for ind, (root, dirs, files) in enumerate(os.walk(input_directory)):
        # estimate at which level we are
        depth = len(pathlib.Path(root).resolve().parents) - initial_depth

        ###
        # extract rules for this level!
        ###
        currentdepth_rules = RULES_SET[depth]

        ###
        # 1.check whether the "mandatory Folders" are present at this level
        #
        ###
        if currentdepth_rules["mandatory_folders"]:
            # loop over rules, each rule corresponding to one mandatory folder
            for current_mandatoryfolder_rule in currentdepth_rules[
                    "mandatory_folders"]:
                # create the list of regexp
                list_of_mandatory_folders = build_rule_regexp(current_mandatoryfolder_rule)
                for mandatory_folders in list_of_mandatory_folders:
                    dir_res = [search(mandatory_folders, d) is None for d in dirs]
                    if all(dir_res):
                        error_list.append(
                            "Mandatory folder not found for this rule : {}".
                            format(current_mandatoryfolder_rule))

        ###
        # 2. check whether the rules are followed for the folders at this level ("authorized folders")
        #
        ###
        for folder in dirs:
            folder_errs = [
                search(authorized_folders_rule, folder) is None
                for authorized_folders_rule in build_rule_regexp(
                    currentdepth_rules["authorized_folders"])
            ]
            # if none of the authorized rules is respected, raise an error
            if all(folder_errs):
                error_list.append("Naming rule not respected for this directory : {}"
                                  "".format(op.join(root, folder)))

        ###
        # 3. check whether rules are followed for files within the folder at this level ("authorized data and
        #    metadata files")
        ###
        for current_file in files:
            file_res = list()
            for rules in currentdepth_rules["authorized_metadata_files"]:
                file_res.extend([
                    search(authorized_metadata_file_rule, current_file) is None
                    for authorized_metadata_file_rule in build_rule_regexp(rules)
                ])

            for rules in currentdepth_rules["authorized_data_files"]:
                file_res.extend([
                    search(authorized_data_file_rule, current_file) is None
                    for authorized_data_file_rule in build_rule_regexp(rules)
                ])
                # if none of the authorized rules is respected, raise an error

            if all(file_res):
                error_list.append("Naming rule not respected for this file : {}".format(current_file))

        ###
        # 4. check whether the "mandatory files" are actually present at this level!
        ###
        if len(currentdepth_rules["mandatory_files"]) > 0:
            # loop over rules, each rule corresponding to one mandatory file
            for current_mandatoryfolder_rule in currentdepth_rules["mandatory_files"]:
                list_of_mandatory_folders = build_rule_regexp(current_mandatoryfolder_rule)
                for mandatory_files in list_of_mandatory_folders:
                    file_res = [search(mandatory_files, file) is None for file in files]
                    if all(file_res):
                        error_list.append(
                            "Mandatory file not found for this rule : {}".
                            format(current_mandatoryfolder_rule))

    # if there are no errors, the data set is valid!
    valid = len(error_list) == 0

    return valid, error_list


def search(rules, where):
    return re.compile(rules).search(where)


def build_rule_regexp(rules):
    """
    Function that create rules base on the regexp in rulesStructured.py

    Parameters
    ----------

        rules: list
            list of filenames
            
    Returns
    -------
        list_of_rules : list
            Concatenate every extension with is respective file

    Examples
    -------
    if :
    file_name_regexp = [
        ['sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)([\\w\\-]*)_ephys'],
        ]
    then ,
    list_of_rules = [
        'sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)([\\w\\-]*)_ephys.tsv'
        ]
    """

    list_of_rules = list()
    if len(rules) == 1:
        list_of_rules = rules
    else:
        for filename in rules[:-1]:
            for extension in rules[-1]:
                list_of_rules.append(filename + str(extension))

    return list_of_rules


def main():
    """
    Main file of the AnDOChecker.

    usage: AnDOChecker.py [-h] [-v] path

            positional arguments:
            directory      Name of the directory that contains the data set to be checked

            optional arguments:
            -h, --help     show this help message and exit
            -v, --verbose  increase output verbosity
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="increase output verbosity")
    parser.add_argument("directory",
                        help="Name of the directory to be checked")
    args = parser.parse_args()

    try:
        directory = args.directory
    except IndexError:
        directory = "."

    if not directory:
        print(f"Directory does not exist: {directory}")
        exit(1)
    dataset_validity, error_list = is_valid(directory)
    if dataset_validity:
        print("Congratulations!\n"
              f"{directory} respects the BIDS-animal-ephys specifications")
    else:
        print("Attention!\n" 
              f"{directory} does not respect the BIDS-animal-ephys specifications")
        if args.verbose:
            print("\nHere are the errors that have been identified:")
            for error_message in error_list:
                print("  " + error_message)


if __name__ == "__main__":
    main()
