# Event Extraction Model

The Event Extraction Model converts webpage DOM node data into structured event records. The project combines a DOM-aware boundary detector with a field classifier to extract event-level attributes such as `Name`, `Date`, `Time`, `Location`, `Description`, and `Price` from page content that has already been transformed to CSV.

Key Sections:
- Architecture
- Repo Layout
- Installation
- Label Merging
- Training
- Inference

## Overview

The pipeline operates in two stages:

1. A DOM-aware sequence model reads nodes in rendering order and predicts BIO tags (`O`, `B`, `I`) to find event boundaries.
2. A field classifier labels the nodes inside those event spans with corresponding event fields.

The final output is a JSON array of flat event records with repeat-safe keys such as `Date_1`, `Location_2`, and `Time_1`. The output also includes each event's source and it's number withing that source.

## Dataset Snapshot

The repo already contains a compiled training dataset in [data/full_data.csv](http://github.com/nasrAnthony/Event-Extraction-Model/blob/main/data/full_data.csv).

- Total DOM nodes: `3100`
- Sources: `15`
- Events: `165`
- Field-labeled nodes after Label Merging: `662`
- Mergred fields: `Date`, `Description`, `Location`, `Name`, `Price`, `Time`

Label distribution:

| Label | Count |
| --- | ---: |
| Date | 242 |
| Location | 139 |
| Name | 120 |
| Time | 116 |
| Description | 27 |
| Price | 18 |

## Architecture

### 1. DOM Boundary Extractor

The boundary extractor in [models/dom_extractor.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/models/dom_extractor.py) is the primary model in the pipeline.

- Text encoder: `distilbert-base-uncased`
- Page encoder: Transformer encoder over DOM node sequence
- Prediction target: BIO tags for each node
- Features fused into each node representation:
  - DistilBERT `[CLS]` embedding for `text_context`
  - HTML `tag` embedding
  - `parent_tag` embedding
  - Numeric DOM features
  - Boolean DOM/text features

Training and evaluation logic lives in [train_dom_extractor.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/train_dom_extractor.py).

### 2. Field Classifier

The field classifier in [models/field_classifier.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/train_dom_extractor.py) runs on nodes predicted as part of an event.

- Model: `GradientBoostingClassifier`
- Text features: TF-IDF with up to 300 features and 1-2 gram vocabulary
- Structural features:
  - Numeric DOM features from `config.yaml`
  - Boolean DOM/text features from `config.yaml`
  - One-hot encoded `tag`
  - One-hot encoded `parent_tag`

### 3. Event Assembly

The inference pipeline in [inference.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/inference.py) performs the final assembly:

- Loads the DOM checkpoint and field-classifier bundle.
- Validates and sorts input rows by `source` and `rendering_order`.
- Predicts BIO probabilities for each page.
- Selects event starts using thresholded peak picking.
- Labels event nodes with field names.
- Builds final event spans and serializes flat JSON records.

## Repository Layout

```text
Event-Extraction-Model/
├── config.yaml                  # Model, training, inference, and feature configuration
├── train_dom_extractor.py       # Boundary model training script
├── field_classifier.py          # Classifier model training script
├── inference.py                 # Reusable inference functions
├── requirements.txt             # Pinned Python dependencies
├── data/
│   ├── raw/                     # Original labeled source files
│   ├── cleaned/                 # Cleaned versions of the raw source files
│   └── full_data.csv            # Combined training dataset
├── helpers/
│   ├── clean_data.py            # CSV cleaning utility
│   ├── concat.py                # Dataset concatenation utility
│   ├── dataset.py               # Label normalization, page dataset, collation
│   ├── losses.py                # BIO loss functions
│   ├── metrics.py               # Boundary decoding and evaluation metrics
│   ├── train_utils.py           # DataLoader and epoch helpers
│   └── utils.py                 # Config loading and numeric feature statistics
├── models/
│   ├── dom_extractor.py         # DOM-aware event extractor model
│   ├── field_classifier.py      # Field classifier training script
│   ├── classifier_model.py      # Separate CatBoost baseline experiment
│   └── catboost_info/           # CatBoost training artifacts
├── field_classifier_v1.joblib   # Exported field-classifier bundle checked into the repo
├── predicted_events.json        # Sample serialized prediction output
└── *.ipynb                      # Notebooks used for analysis and experimentation
```

## Installation

The DOM boundary model uses PyTorch and runs on CUDA when available, otherwise on CPU. The code selects the device automatically.

The first run of the boundary model downloads the Hugging Face tokenizer and model weights for `distilbert-base-uncased` unless they are already cached.

To cache tokenizer and model, see details in [test_platform.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/test_platform.py)

## Configuration

All core settings are in [config.yaml](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/config.yaml).

## Data Pipeline

### Raw And Cleaned Source Files

The dataset is stored source-by-source, and the following cleaning is done (as a sanity check more than anything)

- Removing exact duplicate rows
- Stripping whitespace from key string columns
- Casting `link` to string dtype

[helpers/concat.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/helpers/concat.py) concatenates all cleaned source files into a single training file and injects a `source` column from the filename stem.


## Label Merging

The project merges label variants in [helpers/dataset.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/helpers/dataset.py) before model training. Any added labels should be inroduced here before retraining.

Merged label mapping:

| Original Labels | Normalized Label |
| --- | --- |
| `Name`, `NameLink`, `NameLocation` | `Name` |
| `Date`, `DateTime` | `Date` |
| `Time`, `StartTime`, `EndTime`, `StartEndTime`, `TimeLocation` | `Time` |
| `Location` | `Location` |
| `Price` | `Price` |
| `Description`, `Desc`, `Details` | `Description` |
| Any unmapped label | `Other` |

## Input Details

`load_data_and_prepare()` in [inference.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/inference.py) validates the inference input against the checkpoint and the current config.

Required core columns:

- `rendering_order`
- `text_context`
- `tag`
- `parent_tag`

Required numeric features:

- `depth`
- `sibling_index`
- `children_count`
- `same_tag_sibling_count`
- `same_text_sibling_count`
- `text_length`
- `word_count`
- `letter_ratio`
- `digit_ratio`
- `whitespace_ratio`
- `attribute_count`

Required boolean features:

- `has_link`
- `link_is_absolute`
- `parent_has_link`
- `is_leaf`
- `contains_date`
- `contains_time`
- `starts_with_digit`
- `ends_with_digit`
- `has_class`
- `has_id`
- `attr_has_word_name`
- `attr_has_word_date`
- `attr_has_word_time`
- `attr_has_word_location`
- `attr_has_word_link`
- `text_has_word_name`
- `text_has_word_date`
- `text_word_time`
- `text_word_description`
- `text_word_location`

Additional rules:

- `source` is optional during inference. If it is missing, the loader derives it from the CSV filename or from the explicit `source_name` argument.
- Rows are sorted by `source` and `rendering_order` before inference.
- The current pipeline does not use `attributes`, `link`, `parent_index`, `text_word_am`, or `text_word_pm`.

## Training

### Train The DOM Boundary Model

Run: [train_dom_extractor.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/train_dom_extractor.py) or the corresponding train-dom-extractor notebook

What it does:

- Loads `data/full_data.csv`
- Normalizes field labels
- Builds binary event membership and BIO start labels
- Splits data by `source` using `GroupShuffleSplit`

Cross Validation can be skipped if using a previously found threshold. The current model runs using `0.1238`
- Performs source-level cross-validation on the training partition
- Sweeps boundary thresholds to maximize F1

- Retrains on the full training partition
- Evaluates on a holdout test partition
- Saves `models/dom_extractor_checkpoint.pt`

The checkpoint includes:

- Model weights
- Label vocabulary
- HTML tag vocabularies
- Numeric feature means and standard deviations
- Feature-column lists
- Best threshold from cross-validation
- Full config used for the run

### Train The Field Classifier

This is more in line with standard ML model building:

- Loads `data/full_data.csv`
- Applies label normalization
- Filters out `Other`
- Encodes the six target field classes
- Builds TF-IDF, numeric, boolean, tag, and parent-tag features
- Evaluates with a train/test split and optional K-fold cross-validation
- Saves `models/field_classifier.joblib`

The saved bundle includes:

- Fitted `GradientBoostingClassifier`
- Fitted `TfidfVectorizer`
- `LabelEncoder`
- Tag and parent-tag column definitions
- Numeric and boolean feature-column lists
- Feature toggles for `use_tag` and `use_parent_tag`

### Artifacts

The repository currently contains:

- [field_classifier_v1.joblib](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/field_classifier_v1.joblib), an exported field-classifier bundle
- [sample_output.json](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/sample_output.json), a sample prediction export

Fresh training using the script writes new artifacts into `models/`, while using the notebooks saves them outside with the existing ones.

For access to the dom_extractor_checkpoint.pt, see this [Google Drive link](https://drive.google.com/drive/folders/1XeGcXWeU4IIPZ4d4pPaHuoMRtOzptzXk?usp=sharing)

## Inference

Inference is exposed as reusable Python functions in [inference.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/inference.py). The committed code does not define a standalone CLI entry point, but an end-to-end example can be found in [test_platform.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/test_platform.py)

### Inference Output Contract

`predict_events()` returns a list of dictionaries. Each dictionary contains:

- `source`
- `event_number`
- One or more extracted field entries named as `<Field>_<Index>`

Repeated fields are separated by index. This preserves every extracted node instead of collapsing duplicate field labels into a single value.

## Evaluation

### Boundary Evaluation

Boundary evaluation logic is implemented in [helpers/metrics.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/helpers/metrics.py).

The boundary model is scored with:

- Precision
- Recall
- F1

Decoding details:

- The model predicts BIO probabilities for every valid node on the page.
- Candidate event starts are selected from the `B` probability stream.
- Threshold search sweeps values from `0.01` to `0.199` in `0.001` increments.
- Peak selection keeps the leftmost local maximum inside each above-threshold region.
- `min_gap` enforces spacing between consecutive event starts.
- `tol=1` allows a predicted start to match a true start within one node.

### Field Classifier Evaluation

The field classifier reports:

- Full classification report
- Micro F1
- Macro F1
- Confusion matrix

## Additional Baseline

[models/classifier_model.py](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/models/classifier_model.py) contains a separate CatBoost-based binary event classifier experiment. It is not part of the main two-stage extraction pipeline.

That script:

- Predicts `is_event` at node level
- Uses `tag` as categorical input
- Uses `text_context` as text input
- Splits pages by `source`
- Saves a CatBoost model to `classifier.cbm`

`catboost` is required for this script and is not included in the current `requirements.txt`.

## Notebooks

The repository includes notebooks that mirror the scripted workflows and are easier to use for model exploration:

- [data.ipynb](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/data.ipynb): for raw data visualization
- [train-dom-extractor.ipynb](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/train-dom-extractor.ipynb): to train the boundary model
- [field-classifier.ipynb](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/field-classifier.ipynb): to train the classifier model
- [inference.ipynb](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/inference.ipynb): for running the inference testing
- [test-playground.ipynb](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/test-playground.ipynb): where the bulk of the testing was done
- [models/train.ipynb](https://github.com/nasrAnthony/Event-Extraction-Model/blob/main/models/train.ipynb): early standalone test area

## Summary

This repository defines a complete event-extraction workflow for DOM-derived webpage data:

- Data cleaning and dataset assembly
- DOM-aware boundary detection
- Field-level node classification
- Structured JSON event generation
- Reproducible training configuration and saved artifacts
