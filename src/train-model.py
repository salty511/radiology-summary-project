import pandas as pd
from datasets import Dataset
import datasets
import transformers
from transformers import AutoTokenizer
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import numpy as np

def load_dataset_from_df(df: pd.DataFrame):
    """Returns HuggingFace Dataset object from Pandas DataFrame"""

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(0.2)
    dataset_temp = dataset["train"].train_test_split(0.2)
    dataset["train"] = dataset_temp["train"]
    dataset["validation"] = dataset_temp["test"]

    return dataset

def batch_process_tokenize_function(batch: datasets.formatting.formatting.LazyBatch, tokenizer: transformers.models.t5.tokenization_t5.T5Tokenizer):
    """Returns HuggingFace Datasets batch dictionary after processing and tokenization"""
    print(tokenizer)
    prefix = "summarize: " # T5 models require "summarize: " prefix
    inputs = [prefix + x for x in batch["FINDINGS"]]
    processed_batch = tokenizer(inputs, max_length=512, truncation=True)

    labels = tokenizer(text_target=batch["IMPRESSION"], max_length=128, truncation=True) # text_target specifies that text input is a target label
    processed_batch["labels"] = labels["input_ids"]
    
    return processed_batch

def tokenize_dataset(dataset: Dataset, tokenizer: transformers.models.t5.tokenization_t5.T5Tokenizer):
    """Returns Dataset object after tokenization and preprocessing, applies batch_process_tokenize_function to each batch via Dataset.map"""

    tokenized_dataset = dataset.map(batch_process_tokenize_function, batched=True, load_from_cache_file=False, fn_kwargs={"tokenizer": tokenizer})
    tokenized_dataset = tokenized_dataset.remove_columns(['IMPRESSION', 'FINDINGS', '__index_level_0__']) # Remove old columns from raw dataset

    return tokenized_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions[predictions == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def load_trainer(tokenizer: transformers.models.t5.tokenization_t5.T5Tokenizer, model_name: str, tokenized_dataset: Dataset):

    batch_size = 16

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    output_dir = f"checkpoints/{model_name}-finetuned/all-raw-data"

    args = Seq2SeqTrainingArguments(
        output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        num_train_epochs=5,
        predict_with_generate=True,
        bf16=True,
        generation_max_length=128,
        push_to_hub=False,
        save_strategy="epoch",
        metric_for_best_model="eval_rouge1",
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    return trainer
    
def test_model(model_path: str, dataset: Dataset):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    findings = dataset["test"]["FINDINGS"]
    original_impressions = dataset["test"]["IMPRESSION"]
    model_impressions = []

    for x in findings:
        prompt = f"Findings: {x}\nImpression: "
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**input_ids)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        model_impressions.append(text)

    from rouge import Rouge
    rouge = Rouge()
    instruct_model_results = rouge.get_scores(
        model_impressions,
        list(original_impressions), avg=True)
    
    print('Model Results on Test Dataset:')
    print(instruct_model_results)


if __name__ == "__main__":
    data_file = "processed_data_no_redacted_info"

    df = pd.read_csv(f'data/{data_file}.csv', sep='$')
    df = df.dropna()

    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    dataset = load_dataset_from_df(df)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Print some info about dataset after processing

    print(f"Shapes of the datasets after processing:")
    print(f"Training: {tokenized_dataset["train"].shape}")
    print(f"Validation: {tokenized_dataset["validation"].shape}")
    print(f"Test: {tokenized_dataset["test"].shape}")
    print(tokenized_dataset)

    metric = load("rouge")

    trainer = load_trainer(tokenizer, model_name, tokenized_dataset)

    trainer.train()

    trainer.save_model(f"models/{model_name}-finetuned/no-redacted-data")

    test_model(f"models/{model_name}-finetuned/no-redacted-data", dataset)
