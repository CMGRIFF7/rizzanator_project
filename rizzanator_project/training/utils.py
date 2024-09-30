# utils.py

def compute_metrics(pred):
    from datasets import load_metric
    rouge = load_metric('rouge')
    predictions, labels = pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        'rouge1': result['rouge1'].mid.fmeasure,
        'rouge2': result['rouge2'].mid.fmeasure,
        'rougeL': result['rougeL'].mid.fmeasure,
    }
