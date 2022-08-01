from datasets import Dataset, DatasetDict
from pathlib import Path
import pandas as pd

# Explicit mapping from text labels to numeric labels
label_dict = {
    "BACKGROUND": 0,
    "OBJECTIVE": 1,
    "METHODS": 2,
    "RESULTS": 3,
    "CONCLUSIONS": 4
}


def load_helper(filename, linenumber=False, cutoff = None, group_by_abstracs = False):

    add_func = (lambda global_list, local_list: global_list.append(local_list)) if group_by_abstracs else \
               (lambda global_list, local_list: global_list.extend(local_list))

    # Lists to temporarily store values
    labels, sentences, abstract_ids, linenumbers = [], [], [], []

    # Abstract Id indicates is indicated at the beginning of each block of
    # sentences. This needs to be stored in a local variable.
    abstract_id = None

    abstract_lines = ""

    # Count number of samples
    count = 0

    # Iterate over all lines in the file in order
    for line in open(filename).readlines():

        # Lines with format "###<abstract_id>" indicate a new abstract
        if line[:3] == "###":
            abstract_id = int(line[3:])
            abstract_lines = "" # reset abstract string

        elif line.isspace():  # check to see if line is a new line
            abstract_line_split = abstract_lines.splitlines()  # split abstract into separate lines
            
            # Iterate through each line in abstract and count them at the same time.
            # Lines are assumed to be in the format "<label>\t<text>"
            labels_local, sentences_local, abstract_ids_local, linenumbers_local = [], [], [], []
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                target_text_split = abstract_line.split("\t")  # split target label from text
                labels_local.append(label_dict[target_text_split[0]])
                sentences_local.append(target_text_split[1])
                abstract_ids_local.append(abstract_id)

                if linenumber:
                    linenumbers_local.append(abstract_line_number / (len(abstract_line_split) - 1))

                # Count number of samples
                count += 1
            
            # Add local results to global results
            add_func(labels, labels_local)
            add_func(sentences, sentences_local)
            add_func(abstract_ids, abstract_ids_local)
            add_func(linenumbers, linenumbers_local)

        else:  # if the above conditions aren't fulfilled, the line contains a labelled sentence
            abstract_lines += line

        # Cutoff if count is bigger than cutoff
        if not (cutoff is None) and count > cutoff:
            break
    
    # Return a dict of lists
    result = {
        "label": labels,
        "sentence": sentences,
        "abstract_id": abstract_ids
    }

    if linenumber:
        result["line_relative"] = linenumbers

    return result


def load_data(data_dir = Path("./data"), linenumber = False, cutoff = None, group_by_abstracs = False):

    dd = {
        name: Dataset.from_dict(load_helper(data_dir.joinpath(name + ".txt"), linenumber, cutoff, group_by_abstracs))
        for name in ["train", "dev", "test"]
    }
    return DatasetDict(dd)


def load_data_as_dataframe(data_dir = Path("./data"), linenumber = False, cutoff = None, group_by_abstracs = False):

    train_data = pd.DataFrame.from_dict(load_helper(data_dir.joinpath("train" + ".txt"), linenumber, cutoff, group_by_abstracs))
    dev_data = pd.DataFrame.from_dict(load_helper(data_dir.joinpath("dev" + ".txt"), linenumber, cutoff, group_by_abstracs))
    test_data = pd.DataFrame.from_dict(load_helper(data_dir.joinpath("test" + ".txt"), linenumber, cutoff, group_by_abstracs))

    return train_data, dev_data, test_data
