#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：05/08/2022 9:57 
# ====================================
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AlbertTokenizer
from models.collate import collate_fn
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.tager import SpanLabel, RelationLabel

def run_test(model_path, test_path, bert_model, output_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {device}")
    
    # Periksa apakah test_path adalah direktori atau file
    if os.path.isdir(test_path):
        # Jika direktori, buat path untuk file dev_triplets.txt
        test_file_path = os.path.join(test_path, "dev_triplets.txt")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"File dev_triplets.txt tidak ditemukan di {test_path}")
    else:
        # Jika sudah berupa file, gunakan langsung
        test_file_path = test_path
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"File test {test_file_path} tidak ditemukan")
    
    # tokenizer
    print(f"Loading tokenizer from {bert_model}")
    # Gunakan AlbertModel khusus untuk indobert-lite-base-p2
    if "indobert-lite" in {bert_model}:
        print(f"Using AlbertModel for {bert_model}")
        tokenizer = AlbertTokenizer.from_pretrained(bert_model)
    else:
        print(f"Using BertModel for {bert_model}")
        tokenizer = BertTokenizer.from_pretrained(bert_model)
    
    #tokenizer = BertTokenizer.from_pretrained(bert_model)

    print("Building SPAN-ASTE model...")
    # get dimension of target and relation
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
    # build span-aste model
    model = SpanAsteModel(
        bert_model,
        target_dim,
        relation_dim,
        device=device
    )

    print(f"Loading model from {model_path}")
    model_file_path = os.path.join(model_path, "model.pt")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file tidak ditemukan di {model_file_path}")
    
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    print(f"Processing test data from {test_file_path}")
    with open(test_file_path, "r", encoding="utf8") as f:
        data = f.readlines()
    
    res = []
    for d in data:
        text, label = d.strip().split("####")

        tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]

        input = tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors="pt")

        input_ids = input.input_ids.to(device)
        attention_mask = input.attention_mask.to(device)
        token_type_ids = input.token_type_ids.to(device)
        seq_len = (input_ids != 0).sum().item()

        # forward
        spans_probability, span_indices, relations_probability, candidate_indices = model(
            input_ids, attention_mask, token_type_ids, [seq_len])

        relations_probability = relations_probability.squeeze(0)
        predict = []
        for idx, can in enumerate(candidate_indices[0]):
            a, b, c, d = can
            aspect = tokenizer.convert_tokens_to_string(tokens[a:b+1])  # Include endpoint
            opinion = tokenizer.convert_tokens_to_string(tokens[c:d+1])  # Include endpoint
            sentiment = RelationLabel(relations_probability[idx].argmax(-1).item()).name

            if sentiment != RelationLabel.INVALID.name:
                predict.append((aspect, opinion, sentiment))
        print("Text:", text)
        print("Predict:", predict)
        
        labels = []
        words = text.split(" ")
        for l in eval(label):
            a, o, sm = l
            a = " ".join([words[i] for i in a])
            o = " ".join([words[i] for i in o])
            labels.append((a, o, sm))
        print("Label:", labels)
        print("-" * 80)
        
        res.append({"text": text, "predict": predict, "label": labels})

    # Set default output path if not provided
    if output_path is None:
        # Extract folder name dari test_path untuk nama file
        if os.path.isdir(test_path):
            folder_name = os.path.basename(os.path.normpath(test_path))
        else:
            folder_name = os.path.basename(os.path.dirname(test_path))
        output_path = f"test_results_{folder_name}.xlsx"
    
    print(f"Saving results to {output_path}")
    dataframe = pd.DataFrame(res)
    dataframe.to_excel(output_path)
    print("Testing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a SPAN-ASTE model on a dataset")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the directory containing the model.pt file")
    parser.add_argument("--test_path", type=str, required=True, 
                        help="Path to the test data folder (will use dev_triplets.txt inside) or direct path to test file")
    parser.add_argument("--bert_model", type=str, default="indobenchmark/indobert-base-p2", 
                        help="BERT model to use (default: indobenchmark/indobert-base-p2)")
    parser.add_argument("--output_path", type=str, default=None, 
                        help="Path to save the output Excel file (default: test_results_<foldername>.xlsx)")
    
    args = parser.parse_args()
    
    run_test(args.model_path, args.test_path, args.bert_model, args.output_path)