import time
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, set_seed
from peft import get_peft_model, LoraConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
import json
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
parser = argparse.ArgumentParser(prog="Depression Classifier Training Script")
parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gradient_norm_value", type=float, default=1.0)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--use_peft", action="store_true")
parser.add_argument("--save", action="store_true")
args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_name = f"[FT][D] {args.model_name.split('/')[1]}{'[QT]' if args.use_quantization else ''}"
wandb.init(
    project="depression_detection",
    name=run_name,
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_norm": args.gradient_norm_value,
        "quantization": args.use_quantization,
        "peft": args.use_peft,
        "patience": args.patience,
        "runs": args.runs,
    },
)

# --- Tokenizer loading
model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "right"

# --- Dataset loading
dw_dataset = prepare_daic_woz("./data/DAIC-WOZ")
print(dw_dataset)


def collate_fn(batch):
    tensors = tokenizer(
        [example["dialogue"] for example in batch],
        padding=True,
        return_tensors="pt",
    )

    labels = [example["symptoms"] for example in batch]
    return {"dialogue": tensors["input_ids"], "labels": labels}


train_dataloader = DataLoader(
    dw_dataset["train"],
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    shuffle=True,
)

validation_dataloader = DataLoader(
    dw_dataset["valid"],
    batch_size=args.batch_size,
    collate_fn=collate_fn,
)

test_dataloader = DataLoader(
    dw_dataset["test"],
    batch_size=args.batch_size,
    collate_fn=collate_fn,
)

# ---- Training
seeds = [42, 12345, 9876, 2024, 8675309]

dep_classif_evals = []
phq_score_evals = []
per_symptom_evals = []
severity_class_evals = []

raw_outputs = {}

# --- Multi-run Train/Eval to confirm model training
for run in range(args.runs):
    set_seed(seeds[run])
    best_val_loss = float("inf")

    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )

    if args.use_peft:
        # peft_config = LoraConfig(
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     r=64,
        #     bias="none",
        #     task_type="SEQ_CLS",
        #     target_modules=["query_proj", "value_proj"],
        # )
        # model = get_peft_model(model, peft_config)
        pass
    else:
        for param in model.parameters():
            param.requires_grad = False

    # ---- Classifier definition
    class DepressionClassifier(nn.Module):
        def __init__(self):
            super(DepressionClassifier, self).__init__()
            self.fc1 = nn.Linear(model.config.hidden_size, 256)
            self.fc2 = nn.Linear(256, 8)
            self.relu = nn.ReLU()

        def forward(self, feature_vector):
            x = self.relu(self.fc1(feature_vector))
            return self.fc2(x)

    classifier = DepressionClassifier().to(device)
    wandb.watch(classifier, log="all")

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(" > Trainable params: ", total_params)
    wandb.log({"param_count": total_params})

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2, verbose=True
    )

    patience_counter = 0
    start_time = time.time()

    # ---- Training loop
    for epoch in range(args.epochs):
        epoch_train_loss = 0.0
        classifier.train()
        for batch in train_dataloader:
            optimizer.zero_grad()

            labels = torch.tensor(batch["labels"]).float().to(device)
            dialogue = batch["dialogue"].to(device)

            outputs = model(dialogue, output_hidden_states=True)

            cls_token = outputs.hidden_states[-1][:, 0, :].float()

            preds = classifier(cls_token)

            loss = criterion(preds, labels)
            epoch_train_loss += loss.item()
            wandb.log({"training/loss": loss.item()})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), max_norm=args.gradient_norm_value
            )
            optimizer.step()

        # ---- Validation
        model.eval()
        classifier.eval()

        epoch_validation_loss = 0.0
        binary_dep_preds = []
        binary_dep_refs = []

        phq_score_refs = []
        phq_score_preds = []

        raw_preds = []
        raw_refs = []

        severity_class_preds = []
        severity_class_refs = []

        with torch.no_grad():
            for batch in validation_dataloader:
                labels = torch.tensor(batch["labels"]).float().to(device)
                dialogue = batch["dialogue"].to(device)

                outputs = model(dialogue, output_hidden_states=True)
                cls_token = outputs.hidden_states[-1][:, 0, :].float()
                preds = classifier(cls_token)

                loss = criterion(preds, labels)
                epoch_validation_loss += loss.item()

                raw_preds.extend(preds.tolist())
                raw_refs.extend(labels.tolist())

                binary_dep_preds.extend((torch.sum(preds, dim=1) > 9).int().tolist())
                binary_dep_refs.extend((torch.sum(labels, dim=1) > 9).int().tolist())

                phq_score_preds.extend(torch.sum(preds, dim=1).tolist())
                phq_score_refs.extend(torch.sum(labels, dim=1).tolist())

                severity_class_preds.extend(
                    [
                        get_severity_class(score)
                        for score in torch.sum(preds, dim=1).tolist()
                    ]
                )
                severity_class_refs.extend(
                    [
                        get_severity_class(score)
                        for score in torch.sum(labels, dim=1).tolist()
                    ]
                )

        epoch_train_loss /= len(train_dataloader)
        wandb.log({"training/epoch_train_loss": epoch_train_loss})

        epoch_validation_loss /= len(validation_dataloader)
        wandb.log({"training/epoch_val_loss": epoch_validation_loss})

        # scheduler.step(epoch_validation_loss)
        # wandb.log({"train_lr": optimizer.param_groups[0]["lr"]})

        classif_validation_evals = compute_dep_classif_metrics(
            binary_dep_preds, binary_dep_refs
        )

        phq_score_validation_evals = compute_phq_score_metrics(
            phq_score_preds, phq_score_refs
        )

        per_symptom_validation_evals = calculate_per_symptom_metrics(
            raw_preds, raw_refs
        )

        severity_class_validation_evals = compute_severity_class_evals(
            severity_class_preds, severity_class_refs
        )

        print(json.dumps(classif_validation_evals, indent=4))
        print(json.dumps(phq_score_validation_evals, indent=4))
        # print(json.dumps(per_symptom_validation_evals, indent=4))

        # wandb.log({"validation/": val_metrics})
        # print(json.dumps(val_metrics, indent=4))

        if epoch_validation_loss < best_val_loss:
            best_val_loss = epoch_validation_loss
            if args.use_peft:
                best_model_state = model.state_dict()
            best_cls_head_state = classifier.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f" > Early stopping triggered at epoch {epoch}.")
                break

        print(
            f" > Epoch {epoch + 1}, train loss: {epoch_train_loss:.4f}, valid loss: {epoch_validation_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}, Patience: {patience_counter}/{args.patience}."
        )

    print(f" > Run {run + 1}: Finished Training: {(time.time() - start_time):.2f}s")

    # ---- Evaluation
    if args.use_peft:
        model.load_state_dict(best_model_state)
    classifier.load_state_dict(best_cls_head_state)
    model.eval()
    classifier.eval()

    binary_dep_preds = []
    binary_dep_refs = []

    phq_score_refs = []
    phq_score_preds = []

    raw_preds = []
    raw_refs = []

    severity_class_preds = []
    severity_class_refs = []
    with torch.no_grad():
        classifier.eval()
        for batch in test_dataloader:
            labels = torch.tensor(batch["labels"]).float().to(device)
            dialogue = batch["dialogue"].to(device)

            outputs = model(dialogue, output_hidden_states=True)
            feature_vectors = outputs.hidden_states[-1].float().mean(1)
            preds = classifier(feature_vectors)

            raw_preds.extend(preds.tolist())
            raw_refs.extend(labels.tolist())

            binary_dep_preds.extend((torch.sum(preds, dim=1) > 9).int().tolist())
            binary_dep_refs.extend((torch.sum(labels, dim=1) > 9).int().tolist())

            phq_score_preds.extend(torch.sum(preds, dim=1).tolist())
            phq_score_refs.extend(torch.sum(labels, dim=1).tolist())

            severity_class_preds.extend(
                [
                    get_severity_class(score)
                    for score in torch.sum(preds, dim=1).tolist()
                ]
            )
            severity_class_refs.extend(
                [
                    get_severity_class(score)
                    for score in torch.sum(labels, dim=1).tolist()
                ]
            )

    raw_outputs[f"run_{run}"] = {"refs": raw_refs, "preds": raw_preds}

    print(f" > Run {run + 1} test eval:")

    classif_run_evals = compute_dep_classif_metrics(binary_dep_preds, binary_dep_refs)
    dep_classif_evals.append(classif_run_evals)
    print(json.dumps(classif_run_evals, indent=4))

    phq_score_run_evals = compute_phq_score_metrics(phq_score_preds, phq_score_refs)
    phq_score_evals.append(phq_score_run_evals)
    print(json.dumps(phq_score_run_evals, indent=4))

    per_symptom_run_evals = calculate_per_symptom_metrics(raw_preds, raw_refs)
    print(json.dumps(per_symptom_run_evals, indent=4))
    per_symptom_evals.append(per_symptom_run_evals)

    severity_class_run_evals = compute_severity_class_evals(
        severity_class_preds, severity_class_refs
    )
    print(json.dumps(severity_class_run_evals, indent=4))
    severity_class_evals.append(severity_class_run_evals)

    print(f" > Run {run + 1} execution time {(time.time() - start_time):.2f}s")


print("===== Final metrics =====")
avg_classif_evals = compute_average_metrics(dep_classif_evals)
print(json.dumps(avg_classif_evals, indent=4))
for metric in avg_classif_evals:
    wandb.log({f"classif_eval/avg_{metric}": avg_classif_evals[metric]["score"]})

avg_phq_score_evals = compute_average_metrics(phq_score_evals)
print(json.dumps(avg_phq_score_evals, indent=4))
for metric in avg_phq_score_evals:
    wandb.log(
        {f"phq_regression_eval/avg_{metric}": avg_phq_score_evals[metric]["score"]}
    )

avg_severity_class_evals = compute_average_metrics(severity_class_evals)
print(json.dumps(avg_severity_class_evals, indent=4))
for metric in avg_severity_class_evals:
    wandb.log(
        {f"severity_class_eval/avg_{metric}": avg_severity_class_evals[metric]["score"]}
    )


avg_symptom_metrics = compute_average_symptom_metrics(per_symptom_evals)
print(json.dumps(avg_symptom_metrics, indent=4))
for symptom in avg_symptom_metrics:
    for metric in avg_symptom_metrics[symptom]:
        wandb.log(
            {
                f"per_symptom_eval/{symptom}_avg_{metric}": avg_symptom_metrics[
                    symptom
                ][metric]["score"]
            }
        )

avg_results = {
    "binary_classification": avg_classif_evals,
    "phq_score": avg_phq_score_evals,
    "severity_class": avg_severity_class_evals,
    "per_symptom_metrics": avg_symptom_metrics,
}

save_name = f"{args.model_name.split('/')[1]}_{'peft' if args.use_peft else ''}_deep"
with open(f"./{save_name}_avg_results.json", "w") as json_file:
    json.dump(avg_results, json_file, indent=4)
wandb.save(f"{save_name}_avg_results.json")

with open(f"./{save_name}_raw_outputs.json", "w") as json_file:
    json.dump(raw_outputs, json_file, indent=4)
wandb.save(f"{save_name}_raw_outputs.json")

wandb.finish()
