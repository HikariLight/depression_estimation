from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import json
import time
import argparse
import re
import random
import wandb
from prepare_datasets import prepare_daic_woz
from utils import (
    compute_dep_classif_metrics,
    compute_phq_score_metrics,
    calculate_per_symptom_metrics,
    compute_average_metrics,
    compute_average_symptom_metrics,
    get_severity_class,
    compute_severity_class_evals,
)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Local LLM ICL eval script")
parser.add_argument(
    "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
)
parser.add_argument("--max_length", type=int, default=8192)
parser.add_argument("--few_shots", type=int, default=3)
parser.add_argument("--prompt", type=str, default="few_shot")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
print(args)

# # ---- Hyperparams
# model_name = args.model_name

# # ---- Model loading
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     attn_implementation="flash_attention_2",
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.generation_config.temperature = None
# model.generation_config.top_p = None
# args.verbose and print(model.generation_config)

# ---- Config loading
with open("./prompts.json", "r") as json_file:
    prompts = json.load(json_file)

generation_prompt = prompts[args.prompt]
regularization_prompt = prompts["regularisation"]
args.verbose and print(generation_prompt)

print(generation_prompt)


# def parse_model_output(output: str):
#     pattern = r"\b[A-Z]{3}\s*-\s*(\d)"

#     symptoms = [int(match.group(1)) for match in re.finditer(pattern, output)]
#     symptoms = symptoms[:8]

#     if len(symptoms) != 8:
#         return None

#     return symptoms


# # ---- Util function
# def generate(
#     prompt: str,
#     input: str,
#     examples: str | None = None,
#     max_length=8192,
#     log_length=False,
# ):
#     messages = [{"role": "user", "content": prompt.format(examples, input)}]
#     inputs = tokenizer.apply_chat_template(
#         messages, return_tensors="pt", add_generation_prompt=True
#     ).to("cuda")
#     attention_mask = torch.ones_like(inputs)
#     generated_ids = model.generate(
#         inputs,
#         attention_mask=attention_mask,
#         max_new_tokens=max_length,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=False,
#         num_beams=1,
#     )

#     input_length = inputs.shape[1]
#     total_length = generated_ids.shape[1]
#     generated_length = total_length - input_length

#     print(" > Generated length: ", generated_length)
#     log_length and wandb.log({"inference/generated_tokens": generated_length})

#     output = tokenizer.batch_decode(
#         generated_ids[:, input_length:], skip_special_tokens=True
#     )[0]
#     return output


# # ---- Dataset prep -----
# dw_dataset = prepare_daic_woz("./data/DAIC-WOZ")
# print(dw_dataset)

# # # ---- Sanity check
# # element = dw_dataset["train"][5]
# # start_time = time.time()
# # model_output = generate(generation_prompt, element["dialogue"])
# # print(model_output)
# # print(" > Parsed: ", parse_model_output(model_output))
# # print(" > Golden: ", element["symptoms"])
# # print(f" > Execution time: {(time.time() - start_time):.2f}s")


# def construct_few_shot_string(few_shot_examples):
#     few_shot_string = ""

#     symptoms = ["LOI", "DEP", "SLE", "ENE", "EAT", "LSE", "CON", "MOV"]

#     for element in few_shot_examples:
#         few_shot_string += f"#### Example dialogue:\n{element['dialogue']}\n\n#### Example dialogue's symptoms:\n"
#         for i, score in enumerate(element["symptoms"]):
#             few_shot_string += f"{symptoms[i]} - {score}\n"

#     return few_shot_string


# # ---- Inference
# seeds = [42, 12345, 9876, 2024, 8675309]
# raw_outputs = {}


# for n in range(1, args.few_shots + 1):
#     run_name = f"[FS][{n}_shot] {args.model_name.split('/')[1]}"
#     run = wandb.init(
#         project="depression_detection",
#         name=run_name,
#         config={
#             "prompt": args.prompt,
#         },
#     )

#     print("-" * 10, f"Evaluation {n}_shot", "-" * 10)

#     dep_classif_evals = []
#     phq_score_evals = []
#     per_symptom_evals = []
#     severity_class_evals = []

#     raw_outputs = {}

#     for seed in seeds:
#         set_seed(seed)

#         print("-" * 5, f"Evaluation seed: {seed}", "-" * 5)

#         # ---- Constructing few_shot examples
#         few_shot_examples = []
#         while True:
#             random_index = random.randint(0, len(dw_dataset["train"]) - 1)
#             random_element = dw_dataset["train"][random_index]
#             if random_element not in few_shot_examples:
#                 few_shot_examples.append(random_element)
#                 break

#         few_shot_string = construct_few_shot_string(few_shot_examples)

#         # ---- Inference
#         phq_score_refs = []
#         phq_score_preds = []

#         binary_dep_refs = []
#         binary_dep_preds = []

#         raw_preds = []
#         raw_refs = []

#         severity_class_preds = []
#         severity_class_refs = []

#         for element in dw_dataset["test"]:
#             model_output = generate(
#                 generation_prompt,
#                 element["dialogue"],
#                 examples=few_shot_string,
#                 max_length=args.max_length,
#             )

#             preds = parse_model_output(model_output)

#             args.verbose and print(model_output)
#             args.verbose and print(" > Preds: ", preds)
#             args.verbose and print(" > Refs", element["symptoms"])

#             if not preds:
#                 print(" > Unparsable output.")

#                 while True:
#                     regularized_output = generate(
#                         regularization_prompt, model_output, log_length=False
#                     )
#                     preds = parse_model_output(regularized_output)
#                     if preds:
#                         args.verbose and print(" > Regularized Preds: ", preds)
#                         break

#             raw_preds.append(preds)
#             raw_refs.append(element["symptoms"])

#             binary_dep_preds.append(int(sum(preds) > 9))
#             binary_dep_refs.append(int(sum(element["symptoms"]) > 9))

#             phq_score_preds.append(sum(preds))
#             phq_score_refs.append(sum(element["symptoms"]))

#             severity_class_preds.append(get_severity_class(sum(preds)))
#             severity_class_refs.append(get_severity_class(sum(element["symptoms"])))

#         # ---- Eval
#         classif_seed_evals = compute_dep_classif_metrics(
#             binary_dep_preds, binary_dep_refs
#         )
#         dep_classif_evals.append(classif_seed_evals)
#         print(json.dumps(classif_seed_evals, indent=4))

#         phq_score_seed_evals = compute_phq_score_metrics(
#             phq_score_preds, phq_score_refs
#         )
#         phq_score_evals.append(phq_score_seed_evals)
#         print(json.dumps(phq_score_seed_evals, indent=4))

#         severity_class_seed_evals = compute_severity_class_evals(
#             severity_class_preds, severity_class_refs
#         )
#         severity_class_evals.append(severity_class_seed_evals)
#         print(json.dumps(severity_class_seed_evals, indent=4))

#         per_symptom_seed_evals = calculate_per_symptom_metrics(raw_preds, raw_refs)
#         per_symptom_evals.append(per_symptom_seed_evals)
#         print(json.dumps(per_symptom_seed_evals, indent=4))

#         raw_outputs[seed] = {"refs": raw_refs, "preds": raw_preds}

#     avg_classif_evals = compute_average_metrics(dep_classif_evals)
#     print(json.dumps(avg_classif_evals, indent=4))
#     for metric in avg_classif_evals:
#         wandb.log({f"classif_eval/avg_{metric}": avg_classif_evals[metric]["score"]})

#     avg_severity_class_evals = compute_average_metrics(severity_class_evals)
#     print(json.dumps(avg_severity_class_evals, indent=4))
#     for metric in avg_severity_class_evals:
#         wandb.log(
#             {
#                 f"severity_class_eval/avg_{metric}": avg_severity_class_evals[metric][
#                     "score"
#                 ]
#             }
#         )

#     avg_phq_score_evals = compute_average_metrics(phq_score_evals)
#     print(json.dumps(avg_phq_score_evals, indent=4))
#     for metric in avg_phq_score_evals:
#         wandb.log(
#             {f"phq_regression_eval/avg_{metric}": avg_phq_score_evals[metric]["score"]}
#         )

#     avg_symptom_metrics = compute_average_symptom_metrics(per_symptom_evals)
#     print(json.dumps(avg_symptom_metrics, indent=4))
#     for symptom in avg_symptom_metrics:
#         for metric in avg_symptom_metrics[symptom]:
#             wandb.log(
#                 {
#                     f"per_symptom_eval/{symptom}_avg_{metric}": avg_symptom_metrics[
#                         symptom
#                     ][metric]["score"]
#                 }
#             )

#     avg_results = {
#         "binary_classification": avg_classif_evals,
#         "phq_score": avg_phq_score_evals,
#         "severity_class": avg_severity_class_evals,
#         "per_symptom_metrics": avg_symptom_metrics,
#     }

#     save_name = f"{args.model_name.split('/')[1]}_{n}_shot"
#     with open(f"./{save_name}_avg_results.json", "w") as json_file:
#         json.dump(avg_results, json_file, indent=4)
#     wandb.save(f"{save_name}_avg_results.json")

#     with open(f"./{save_name}_raw_outputs.json", "w") as json_file:
#         json.dump(raw_outputs, json_file, indent=4)
#     wandb.save(f"{save_name}_raw_outputs.json")

#     wandb.finish()
