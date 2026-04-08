from src.data import run_data_pipeline
from src.test import test_model
import pandas as pd
from transformers import AutoTokenizer
from src.train import load_dataset_from_df, tokenize_dataset, load_trainer

if __name__ == '__main__':
    datafile = "processed_data_clean"

    run_data_pipeline(datafile)

    df = pd.read_csv(f'data/{datafile}.csv', sep='$')

    model_name = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_dataset_from_df(df)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Print some info about dataset after processing

    print(f"Shapes of the datasets after processing:")
    print(f"Training: {tokenized_dataset["train"].shape}")
    print(f"Validation: {tokenized_dataset["validation"].shape}")
    print(f"Test: {tokenized_dataset["test"].shape}")
    print(tokenized_dataset)

    trainer = load_trainer(tokenizer, model_name, tokenized_dataset, datafile)

    trainer.train()

    trainer.save_model(f"models/{model_name.split("/")[1]}-finetuned/{datafile}")

    test_model(f"models/{model_name.split("/")[1]}-finetuned/{datafile}", dataset)
