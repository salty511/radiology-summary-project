# Radiology Text Summarisation Model Project

This Repo contains an end to end modelling pipeline for fine-tuning the t5-small model on a dataset of Radiology scan findings, and corresponding summaries (impressions). See the live demo for a basic web intereface for interacting with the final model.

<a href="https://radiology-summary-project.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/Live%20Demo-Render-blue?style=flat-square&logo=netlify&logoSize=auto" alt="Live Demo Badge" class="border-none">
</a>

## Model Summary

I initially explored the T5 family of models. These are encoder-decoder models designed for sequence to sequence tasks including text summarisation. There are also more modern versions of the original T5 models called flan-t5 which perform better on almost all tasks, and as I've seen on this task as well. I also tried using a quantised version of a larger model flan-t5-large (0.8B parameters), using LoRA for fine-tuning although I got worse results.

For this task I chose to stick with the smaller models for both training and inference speed on consumer hardware. See table below for a comparison of the performance for the various models I tried. Ultimately I went with the flan-t5-small model as it's good enough to achieve the minimum rouge-1 score of 0.3 and small enough to be trainable on my local machine. I chose to use the HuggingFace Transformers framework for finetuning, utilising the Seq2SeqTrainer class.

| Model                           | Eval Rouge-1 | Test Rouge-1 |
| ------------------------------- | ------------ | ------------ |
| t5-small                        | 0.4941       | 0.3892       |
| flan-t5-small                   | 0.5928       | 0.6213       |
| flan-t5-large (8-bit Quantised) | 0.5021       | 0.4012       |

### Training Parameters

I mainly stuck with deafult training parameters, although in future would like to explore some hyperparameter tuning. I did experiment with number of training epochs and landed on 5 as the point where eval rouge-1 starts to converge.

## Dataset and Preprocessing/Cleaning

The dataset consists of many xml files, containing an `<Abstract>` section, with `<AbstractText>` subsections. The sections of interest to us are the ones labelled `#FINDINGS` and `#IMPRESSIONS`. These are the scan findings and corresponding summaries respectively. There are files which do not contian either one or both sections so these are excluded from modelling.

There are 3999 total datafiles in the dataset. After cleaning, these are processed into 1809 total findings, impression pairs for finetuning.

### 1. Redacted Information

The dataset contains some redacted personal information, appearing as strings of Xs in the text, see an example of an impression containing this feature below. I chose to try modelling both with and without these records, and ultimately achieved better results with simply removing these records from the dataset.

    Comparison XXXX, XXXX. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified.

### 2. Impression Formatting

The impressions in the dataset consist of two distinct formats, either a simple sentenct (or string of sentences) or a numbered list of sentences. See examples below.

    1. Focal opacity in the right midlung zone worrisome for pneumonitis. 2. Mild pulmonary vascular congestion.
    No acute cardiopulmonary findings

I chose to standardise all records to the simple sentences format.

### 3. Other Artefacts

There are also some other small artefacts in the dataset, for example some records contain trailing full stops at the end of sentences, see an example below

    Surgical changes of the right hemithorax and mild cardiomegaly without acute cardiopulmonary abnormality identified. .

There's also inconsistencies in whether sentences end with a full stop or not, some records are just the text of the sentence with no full stop. I chose to standardise these and remove the trailing fullstops as seen above.

See table below for a comparison of rouge score on the flan-t5-small model after each step of data cleaning.

| Dataset      | Eval Rouge-1 | Test Rouge-1 |
| ------------ | ------------ | ------------ |
| All Raw Data | 0.4265       | 0.3680       |
| Step 1       | 0.5921       | 0.5491       |
| Step 2       | 0.6185       | 0.5472       |
| Step 3       | 0.5928       | 0.6213       |

Interestingly, we get a large gain from removing the redacted information, but subsequent steps seem to make the model slightly worse. Ultimately they are all quite close though and I think the difference is largely just due to random variation.

## Usage

### Data

The data can be found at `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz`

Extract the `ecgen-radiology` folder to `data/` in the root directory

### Install Requirements

```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
python main.py
```

This script runs the data preprocessing and cleaning and trains the model, then calculates the rouge score on the test dataset. The model is output to `models/flan-t5-small-finetuned/processed_data_clean/`

### Run the API

```bash
cd api
fastapi dev
```

Open `http://127.0.0.1:8000/` to use the web form, or `http://127.0.0.1:8000/health` for the health check endpoint.

## API Docs

Access the impressions endpoint via http://127.0.0.1:8000/impression. See an example curl command below. The API is also deployed at https://radiology-summary-project.onrender.com/.

```bash
curl -X POST "http://localhost:8000/impression/" \
  -H "Content-Type: application/json" \
  -d '{"findingsText":"No acute cardiopulmonary process. Lungs are clear. Heart size normal."}'
```

```bash
curl -X POST "https://radiology-summary-project.onrender.com/impression/" \
  -H "Content-Type: application/json" \
  -d '{"findingsText":"No acute cardiopulmonary process. Lungs are clear. Heart size normal."}'
```

## Limitations and Future Improvements

For this project I chose to keep it simple by limiting myself to training on my local machine. This comes with limitations on the size of model I can use and as I've shown using a quantised version of a larger model was unrealistic for training time and did not yield better results. I think this was due to the implementation of training with LoRA on a quantised model and would need to investigate this futher. Ultimately I would like to use a cloud computing service to try larger models on better hardware in future.

Another anvenue I could explore is hyperparameter tuning, for simplicity I stuck with mostly default settings for training paramets and only tweaked the epoch number.
