"""
Script version of inference.ipynb, but without all the extra visualization/prints
"""
from inference import load_models, load_data_and_prepare, run_dom_extractor, run_field_classifier, predict_events, save_output
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer, ckpt, field_bundle = load_models(
    checkpoint_path="/Users/yhila/Downloads/dom_extractor_checkpoint_v2.pt",
    classifier_path="/Users/yhila/OneDrive/Desktop/UniFiles/Grad/IndustryProj/Event-Extraction-Model/field-classifier.ipynb",
    device=device
)

df = load_data_and_prepare(
    csv_path="/Users/yhila/Downloads/test_data.csv",
    ckpt=ckpt,
)

results = run_dom_extractor(df, model, tokenizer, ckpt, device)

node_labels = run_field_classifier(df, results, field_bundle)

events = predict_events(df, results, node_labels)

save_output(events, "predicted_events.json")