# Radiology Text Summarisation Model Project

This Repo contains an end to end modelling pipeline for fine-tuning the t5-small model on a dataset of Radiology scan findings, and corresponding summaries (impressions)

## Model Summary

I chose to use the t5-small model as it's designed for sequence to sequence tasks including summarisation. The small model is good enough to achieve the minimum rouge-1 score of 0.3 and small enough to be trainable on consumer hardware. I chose to use the HuggingFace Transformers framework for finetuning, utilising the Seq2SeqTrainer class.

## Dataset and Preprocessing

The dataset consists of many xml files, containing an `<Abstract>` section, with `<AbstractText>` subsections. The sections of interest to us are the ones labelled `#FINDINGS` and `#IMPRESSIONS`. These are the scan findings and corresponding summaries respectively. There are files which don't contian either one or both sections so these are excluded from modelling.

### Redacted Information

The dataset contains some redacted personal information, appearing as strings of Xs in the text, i.e. "Comparison XXXX, XXXX. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified." is an example of an impression containing this feature. I chose to try modelling both with and without these records, and ultimately achieved better results with simply removing these records from the dataset.

## Model Performance

I achieved a rouge-1 score of 0.49 on the validation dataset and 0.38 on the test dataset.
