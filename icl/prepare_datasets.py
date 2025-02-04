from datasets import DatasetDict, Dataset, load_dataset
import csv
import re


def prepare_daic_woz(dataset_path):
    splits = {
        "train": load_dataset(
            "csv", data_files=f"{dataset_path}/train_split_Depression_AVEC2017.csv"
        ),
        "valid": load_dataset(
            "csv", data_files=f"{dataset_path}/dev_split_Depression_AVEC2017.csv"
        ),
        "test": load_dataset(
            "csv", data_files=f"{dataset_path}/Detailed_PHQ8_Labels.csv"
        ),
    }

    splits["test"] = splits["test"].rename_column("PHQ_8NoInterest", "PHQ8_NoInterest")
    splits["test"] = splits["test"].rename_column("PHQ_8Depressed", "PHQ8_Depressed")
    splits["test"] = splits["test"].rename_column("PHQ_8Sleep", "PHQ8_Sleep")
    splits["test"] = splits["test"].rename_column("PHQ_8Tired", "PHQ8_Tired")
    splits["test"] = splits["test"].rename_column("PHQ_8Appetite", "PHQ8_Appetite")
    splits["test"] = splits["test"].rename_column("PHQ_8Failure", "PHQ8_Failure")
    splits["test"] = splits["test"].rename_column(
        "PHQ_8Concentrating", "PHQ8_Concentrating"
    )
    splits["test"] = splits["test"].rename_column("PHQ_8Moving", "PHQ8_Moving")
    splits["test"] = splits["test"].rename_column("PHQ_8Total", "PHQ8_Score")

    ellie_regex = r"\((.*?)\)"
    # files_to_filter = [451, 458, 480]  # Don't contain virtual agent turns
    files_to_filter = [451, 458]  # Don't contain virtual agent turns

    test_split = [
        element["Participant_ID"]
        for element in load_dataset(
            "csv", data_files=f"{dataset_path}/full_test_split.csv"
        )["train"]
    ]

    dataset = {}
    for split in splits:
        dataset[split] = []

        for element in splits[split]["train"]:
            if element["Participant_ID"] in files_to_filter:
                continue

            if split == "test" and element["Participant_ID"] not in test_split:
                continue

            file_name = f"{dataset_path}/{element['Participant_ID']}_TRANSCRIPT.csv"

            # ---- Preparing the dialogue
            with open(file_name) as file:
                csv_reader = csv.reader(file)

                dialogue = ""
                for row in csv_reader:
                    if len(row) > 0:
                        convo_turn = row[0].split("\t")
                        speaker, content = convo_turn[2], convo_turn[3]

                        if content == "<sync>":
                            continue

                        if speaker == "Ellie":
                            if re.search(ellie_regex, content):
                                ellie_speech = re.search(ellie_regex, content).group(1)
                                dialogue += "Doctor: " + ellie_speech + "\n"
                            else:
                                dialogue += "Doctor: " + content + ".\n"

                        if speaker == "Participant":
                            dialogue += "Patient: " + content + ".\n"

            data_item = {
                "id": element["Participant_ID"],
                "dialogue": dialogue,
                "symptoms": [
                    int(value) if value is not None else 0
                    for key, value in element.items()
                    if key.startswith("PHQ8_")
                    and key not in ("PHQ8_Binary", "PHQ8_Score")
                ],
            }
            dataset[split].append(data_item)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(dataset["train"]),
            "valid": Dataset.from_list(dataset["valid"]),
            "test": Dataset.from_list(dataset["test"]),
        }
    )

    return dataset
