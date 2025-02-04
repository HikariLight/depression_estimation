from google import genai
from google.genai import types
import argparse
import re
import json
import time
import wandb
import os
from dotenv import load_dotenv
from prepare_datasets import prepare_daic_woz
from utils import (
    compute_dep_classif_metrics,
    compute_phq_score_metrics,
    calculate_per_symptom_metrics,
    compute_severity_class_evals,
    get_severity_class,
)

load_dotenv()

# --- Params parsing
parser = argparse.ArgumentParser(prog="Cloud LLM ICL eval script")
parser.add_argument("--model_name", type=str, default="gemini-2.0-flash-exp")
parser.add_argument("--time_delay", type=int, default=6)
parser.add_argument("--prompt", type=str, default="zero_shot")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)

run_name = f"[{args.prompt}] {args.model_name}"
wandb.init(
    project="depression_detection",
    name=run_name,
    config={
        "prompt": args.prompt,
    },
)

# ---- Model setup
client = genai.Client(vertexai=False, api_key=os.getenv("GOOGLE_API_KEY"))
model_name = args.model_name

# ---- Data loading
dw_dataset = prepare_daic_woz("./data/DAIC-WOZ")
print(dw_dataset)

# ---- Util
with open("./prompts.json", "r") as json_file:
    prompts = json.load(json_file)

generation_prompt = prompts[args.prompt]
regularization_prompt = prompts["regularisation"]


def parse_model_output(output: str):
    pattern = r"\b[A-Z]{3}\s*-\s*(\d)"

    symptoms = [int(match.group(1)) for match in re.finditer(pattern, output)]
    symptoms = symptoms[:8]

    if len(symptoms) != 8:
        return []

    return symptoms


# # ---- Sanity check
# test_element = dw_dataset["valid"][0]
# response = client.models.generate_content(
#     model=model_name,
#     contents=generation_prompt.format(test_element["dialogue"]),
#     config=types.GenerateContentConfig(temperature=0, topK=1),
# )
# print(response)
# print(response.text)
# preds = parse_model_output(response.text)
# # if not preds:
# #     regularized_output = client.models.generate_content(
# #         model=model_name,
# #         contents=generation_prompt.format(test_element["dialogue"]),
# #         config=types.GenerateContentConfig(temperature=0, topK=1),
# #     )
# #     preds = parse_model_output(regularized_output)
# print(" > Parsed: ", preds)
# print(" > Refs: ", test_element["symptoms"])

# ---- Inference
phq_score_preds = []
phq_score_refs = []

binary_dep_refs = []
binary_dep_preds = []

raw_preds = []
raw_refs = []

severity_class_preds = []
severity_class_refs = []

for element in dw_dataset["test"]:
    response = client.models.generate_content(
        model=model_name,
        contents=generation_prompt.format(element["dialogue"]),
        config=types.GenerateContentConfig(temperature=0, topK=1),
    )

    wandb.log(
        {"inference/generated_tokens": response.usage_metadata.candidates_token_count}
    )

    args.verbose and print(response.text)
    preds = parse_model_output(response.text)

    if not preds:
        print(" > Unparsable output.")
        print(response.text)
        print("-" * 20)

    raw_preds.append(preds)
    raw_refs.append(element["symptoms"])

    args.verbose and print(" > Preds: ", preds)
    args.verbose and print(" > Refs", element["symptoms"])

    binary_dep_preds.append(int(sum(preds) > 9))
    binary_dep_refs.append(int(sum(element["symptoms"]) > 9))

    phq_score_preds.append(sum(preds))
    phq_score_refs.append(sum(element["symptoms"]))

    severity_class_preds.append(get_severity_class(sum(preds)))
    severity_class_refs.append(get_severity_class(sum(element["symptoms"])))

    time.sleep(args.time_delay)  # RPM quota


# ---- Eval
classif_evals = compute_dep_classif_metrics(binary_dep_preds, binary_dep_refs)
print(json.dumps(classif_evals, indent=4))
for metric in classif_evals:
    wandb.log({f"classif_eval/avg_{metric}": classif_evals[metric]})

phq_score_evals = compute_phq_score_metrics(phq_score_preds, phq_score_refs)
print(json.dumps(phq_score_evals, indent=4))
for metric in phq_score_evals:
    wandb.log({f"phq_regression_eval/avg_{metric}": phq_score_evals[metric]})

severity_class_evals = compute_severity_class_evals(
    severity_class_preds, severity_class_refs
)
print(json.dumps(severity_class_evals, indent=4))
for metric in severity_class_evals:
    wandb.log({f"severity_class_eval/avg_{metric}": severity_class_evals[metric]})

per_symptom_evals = calculate_per_symptom_metrics(raw_preds, raw_refs)
print(json.dumps(per_symptom_evals, indent=4))
for symptom in per_symptom_evals:
    for metric in per_symptom_evals[symptom]:
        wandb.log(
            {
                f"per_symptom_eval/{symptom}_avg_{metric}": per_symptom_evals[symptom][
                    metric
                ]
            }
        )

results = {
    "binary_classification": classif_evals,
    "phq_score": phq_score_evals,
    "severity_class": severity_class_evals,
    "per_symptom_metrics": per_symptom_evals,
}

save_name = f"{args.model_name}_{args.prompt}"
with open(f"./{save_name}_avg_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
wandb.save(f"{save_name}_avg_results.json")

with open(f"./{save_name}_raw_outputs.json", "w") as json_file:
    json.dump({"refs": raw_refs, "preds": raw_preds}, json_file, indent=4)
wandb.save(f"{save_name}_raw_outputs.json")

wandb.finish()
