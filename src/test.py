from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge import Rouge
from datasets import DatasetDict

def test_model(model_path: str, dataset: DatasetDict):
    """Generates impressions for all findings in test dataset and calculates rouge score"""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    findings = dataset["test"]["FINDINGS"]
    original_impressions = dataset["test"]["IMPRESSION"]
    model_impressions = []

    prefix = "summarize: "

    for x in findings:
        input_ids = tokenizer(prefix + x, return_tensors="pt").to(model.device)
        output = model.generate(**input_ids)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        model_impressions.append(text)


    rouge = Rouge()
    instruct_model_results = rouge.get_scores(
        model_impressions,
        list(original_impressions), avg=True)
    
    print('Model Results on Test Dataset:')
    print(instruct_model_results)
