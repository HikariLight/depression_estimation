import evaluate
import numpy as np

metric_mae = evaluate.load("mae")
metric_mse = evaluate.load("mse")
metric_f1 = evaluate.load("f1")


def compute_dep_classif_metrics(preds, refs):
    results = {}

    results["f1_macro"] = metric_f1.compute(
        predictions=preds, references=refs, average="macro"
    )["f1"]
    results["f1_micro"] = metric_f1.compute(
        predictions=preds, references=refs, average="micro"
    )["f1"]

    return results


def compute_phq_score_metrics(preds, refs):
    results = {}

    results["mae"] = metric_mae.compute(predictions=preds, references=refs)["mae"]

    results["mse"] = metric_mse.compute(predictions=preds, references=refs)["mse"]

    results["rmse"] = np.sqrt(results["mse"])

    mean_references = np.mean(refs)
    results["rrmse"] = (
        results["rmse"] / mean_references if mean_references != 0 else float("inf")
    )

    return results


def compute_severity_class_evals(preds, refs):
    results = {}

    results["f1_macro"] = metric_f1.compute(
        predictions=preds, references=refs, average="macro"
    )["f1"]
    results["f1_micro"] = metric_f1.compute(
        predictions=preds, references=refs, average="micro"
    )["f1"]

    return results


def compute_average_metrics(data):
    result = {}

    metrics = data[0].keys()

    for metric in metrics:
        scores = [element[metric] for element in data]
        avg = np.mean(scores)
        minimum = np.min(scores)
        maximum = np.max(scores)
        stddev = np.std(scores)

        result[metric] = {
            "score": avg,
            "stddev": stddev,
            "min": minimum,
            "max": maximum,
        }

    return result


def calculate_per_symptom_metrics(raw_preds, raw_refs):
    result = {}

    symptoms = ["LOI", "DEP", "SLE", "ENE", "EAT", "LSE", "CON", "MOV"]

    for i, symptom in enumerate(symptoms):
        symptom_preds = [pred[i] for pred in raw_preds]
        symptom_refs = [ref[i] for ref in raw_refs]

        result[symptom] = {
            "mae": metric_mae.compute(
                predictions=symptom_preds, references=symptom_refs
            )["mae"],
            "mse": metric_mse.compute(predictions=symptom_preds, references=symptom_refs)["mse"],
            "f1_macro": metric_f1.compute(
                predictions=symptom_preds, references=symptom_refs, average="macro"
            )["f1"],
            "f1_micro": metric_f1.compute(
                predictions=symptom_preds, references=symptom_refs, average="micro"
            )["f1"]
        }

        result[symptom]["rmse"] =  np.sqrt(result[symptom]["mse"])

        mean_references = np.mean(symptom_refs)
        result[symptom]["rrmse"] = result[symptom]["rmse"] / mean_references if mean_references != 0 else float("inf")

    return result

def compute_average_symptom_metrics(data):
    result = {}

    symptoms = ["LOI", "DEP", "SLE", "ENE", "EAT", "LSE", "CON", "MOV"]

    for symptom in symptoms:
        symptom_evals = [eval[symptom] for eval in data]
        result[symptom] = compute_average_metrics(symptom_evals)

    return result


def get_severity_class(phq_score):
    return 0 if phq_score < 5 else int(phq_score // 5)
