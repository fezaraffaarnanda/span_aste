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
from transformers import BertTokenizer, BertTokenizerFast
from models.collate import collate_fn
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.tager import SpanLabel, RelationLabel

def merge_subwords(tokens):
    """Menggabungkan token subword menjadi kata-kata utuh"""
    merged_tokens = []
    current_word = ""
    
    for token in tokens:
        # Jika token adalah special token, abaikan
        if token in ['[CLS]', '[SEP]']:
            continue
        # Jika token adalah subword (diawali ##), gabungkan dengan kata sebelumnya
        elif token.startswith('##'):
            current_word += token[2:]  # Ambil karakter setelah ##
        # Jika token normal (bukan subword)
        else:
            # Jika ada kata sebelumnya yang sedang diproses, tambahkan ke hasil
            if current_word:
                merged_tokens.append(current_word)
            # Mulai kata baru
            current_word = token
    
    # Tambahkan kata terakhir yang sedang diproses
    if current_word:
        merged_tokens.append(current_word)
    
    return merged_tokens

def extract_first_word(text):
    """Extract only the first word from a text string, removing special tokens."""
    # Clean special tokens and punctuation
    text = text.replace('[CLS]', '').replace('[SEP]', '').strip()
    # Get only the first word
    words = text.split()
    return words[0] if words else text

def clean_text_thoroughly(text):
    """Thoroughly clean text to remove special tokens and fix punctuation"""
    # Handle various forms of special tokens
    text = text.replace('[SEP]', '').replace('[CLS]', '')
    text = text.replace(' [SEP]', '').replace(' [CLS]', '')
    text = text.replace('[SEP] ', '').replace('[CLS] ', '')
    
    # Remove duplicate spaces
    text = ' '.join(text.split())
    
    # Remove trailing punctuation but preserve internal punctuation
    while text and text[-1] in [',', '.', '!', '?', ';', ':', ' ']:
        text = text[:-1]
    
    return text.strip()

def clean_span(tokens):
    """Clean up token spans by handling any tokenizer type"""
    # Remove special tokens
    filtered_tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]']]
    
    # Skip if empty
    if not filtered_tokens:
        return ""
        
    # Combine tokens into text, handling different subword formats
    text = ""
    for i, token in enumerate(filtered_tokens):
        # Handle BERT-style continuation tokens
        if token.startswith('##'):
            text += token[2:]
        # Handle potential SentencePiece/ALBERT style tokens (often start with ▁)
        elif token.startswith('▁'):
            if i > 0:  # Not the first token
                text += " " + token[1:]
            else:
                text += token[1:]
        # Handle punctuation (don't add spaces before punctuation)
        elif token in [',', '.', '!', '?', ':', ';']:
            text += token
        # Regular token
        else:
            if i > 0:  # Not the first token
                text += " " + token
            else:
                text += token
    
    # Clean up trailing punctuation
    text = text.strip()
    while text and text[-1] in [',', '.', '!', '?']:
        text = text[:-1].strip()
    
    return text

def run_test(model_path, test_path, bert_model, output_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {device}")
    
    # Periksa apakah test_path adalah direktori atau file
    if os.path.isdir(test_path):
        test_file_path = os.path.join(test_path, "dev_triplets.txt")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"File dev_triplets.txt tidak ditemukan di {test_path}")
    else:
        test_file_path = test_path
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"File test {test_file_path} tidak ditemukan")
    
    print(f"Loading tokenizer from {bert_model}")
    if "indobert-lite" in bert_model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
    else:
        from transformers import BertTokenizer
        tokenizer = BertTokenizerFast.from_pretrained(bert_model)
    
    print("Building SPAN-ASTE model...")
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
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
        
        # Menggunakan tokenizer dengan return_offsets_mapping untuk alignment token ke teks asli
        encoded_input = tokenizer(
            text, 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            return_offsets_mapping=True
        )
        input_ids = encoded_input.input_ids.to(device)
        attention_mask = encoded_input.attention_mask.to(device)
        token_type_ids = encoded_input.token_type_ids.to(device)
        offset_mapping = encoded_input.offset_mapping[0].tolist()  # mapping untuk setiap token
        
        # Mendapatkan token asli (dengan special token)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Hitung panjang sekuens (bisa juga menggunakan tokens)
        seq_len = (input_ids != 0).sum().item()
        
        # Forward pass
        spans_probability, span_indices, relations_probability, candidate_indices = model(
            input_ids, attention_mask, token_type_ids, [seq_len]
        )
        
        relations_probability = relations_probability.squeeze(0)
        predict = []
        for idx, can in enumerate(candidate_indices[0]):
            a, b, c, d = can  # indeks dalam token sequence
            # Ambil token untuk aspect dan opinion
            aspect_tokens = tokens[a:b+1]
            opinion_tokens = tokens[c:d+1]
            
            # Gabungkan subword jadi kata utuh
            aspect_words = merge_subwords(aspect_tokens)
            opinion_words = merge_subwords(opinion_tokens)
            
            # Untuk aspect: jika ada lebih dari satu kata, gunakan semua kata kecuali kata terakhir; jika hanya satu, gunakan kata tersebut.
            if len(aspect_words) > 1:
                aspect = " ".join(aspect_words[:-1])
            elif aspect_words:
                aspect = aspect_words[0]
            else:
                aspect = ""
            
            # Untuk opinion: jika ada lebih dari satu kata, gunakan semua kata kecuali kata terakhir; jika hanya satu, gunakan kata tersebut.
            if len(opinion_words) > 1:
                opinion = " ".join(opinion_words[:-1])
            elif opinion_words:
                opinion = opinion_words[0]
            else:
                opinion = ""
            
            # Hilangkan tanda baca di akhir (opsional)
            aspect = aspect.rstrip('.,!?;:')
            opinion = opinion.rstrip('.,!?;:')
            
            # Skip jika salah satu kosong
            if not aspect or not opinion:
                continue
            
            sentiment = RelationLabel(relations_probability[idx].argmax(-1).item()).name
            if sentiment != RelationLabel.INVALID.name:
                predict.append((aspect, opinion, sentiment))
        
        print("Text:", text)
        print("Predict:", predict)
        
        labels = []
        words = text.split(" ")
        for l in eval(label):
            a_idx, o_idx, sm = l
            a_word = " ".join([words[i] for i in a_idx])
            o_word = " ".join([words[i] for i in o_idx])
            labels.append((a_word, o_word, sm))
        print("Label:", labels)
        print("-" * 80)
        
        res.append({"text": text, "predict": predict, "label": labels})
    
    if output_path is None:
        if os.path.isdir(test_path):
            folder_name = os.path.basename(os.path.normpath(test_path))
        else:
            folder_name = os.path.basename(os.path.dirname(test_path))
        output_path = f"test_results_{folder_name}.xlsx"
    
    print(f"Saving results to {output_path}")
    dataframe = pd.DataFrame(res)
    dataframe.to_excel(output_path)
    print("Testing completed successfully!")


# def run_test(model_path, test_path, bert_model, output_path=None):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     print(f"Using device: {device}")
    
#     # Periksa apakah test_path adalah direktori atau file
#     if os.path.isdir(test_path):
#         # Jika direktori, buat path untuk file dev_triplets.txt
#         test_file_path = os.path.join(test_path, "dev_triplets.txt")
#         if not os.path.exists(test_file_path):
#             raise FileNotFoundError(f"File dev_triplets.txt tidak ditemukan di {test_path}")
#     else:
#         # Jika sudah berupa file, gunakan langsung
#         test_file_path = test_path
#         if not os.path.exists(test_file_path):
#             raise FileNotFoundError(f"File test {test_file_path} tidak ditemukan")
    
#     # tokenizer
#     print(f"Loading tokenizer from {bert_model}")

#     if "indobert-lite" in bert_model:
#         print(f"Trying AutoTokenizer for {bert_model}")
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(bert_model)
#     else:
#         print(f"Using BertTokenizer for {bert_model}")
#         tokenizer = BertTokenizer.from_pretrained(bert_model)
    
#     #tokenizer = BertTokenizer.from_pretrained(bert_model)

#     print("Building SPAN-ASTE model...")
#     # get dimension of target and relation
#     target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
#     # build span-aste model
#     model = SpanAsteModel(
#         bert_model,
#         target_dim,
#         relation_dim,
#         device=device
#     )

#     print(f"Loading model from {model_path}")
#     model_file_path = os.path.join(model_path, "model.pt")
#     if not os.path.exists(model_file_path):
#         raise FileNotFoundError(f"Model file tidak ditemukan di {model_file_path}")
    
#     model.load_state_dict(torch.load(model_file_path, map_location=torch.device(device)))
#     model.to(device)
#     model.eval()

#     print(f"Processing test data from {test_file_path}")
#     with open(test_file_path, "r", encoding="utf8") as f:
#         data = f.readlines()
    
    
#     res = []
#     for d in data:
#         text, label = d.strip().split("####")

#         tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
        
#         print("Original text:", text)
#         print("Tokens:", tokens)

#         input = tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors="pt")

#         input_ids = input.input_ids.to(device)
#         attention_mask = input.attention_mask.to(device)
#         token_type_ids = input.token_type_ids.to(device)
#         seq_len = (input_ids != 0).sum().item()

#         # forward
#         spans_probability, span_indices, relations_probability, candidate_indices = model(
#             input_ids, attention_mask, token_type_ids, [seq_len])

#         relations_probability = relations_probability.squeeze(0)
#         predict = []
#         seen_pairs = set()
#         for idx, can in enumerate(candidate_indices[0]):
#             a, b, c, d = can
            
            
#             # original code
            
#             # print(f"Aspect span: {a} to {b}, tokens: {tokens[a:b+1]}")
#             # print(f"Opinion span: {c} to {d}, tokens: {tokens[c:d+1]}")
#             # aspect = tokenizer.convert_tokens_to_string(tokens[a:b+1])  # Include endpoint
#             # opinion = tokenizer.convert_tokens_to_string(tokens[c:d+1])  # Include endpoint
#             # sentiment = RelationLabel(relations_probability[idx].argmax(-1).item()).name

#             # if sentiment != RelationLabel.INVALID.name:
#             #     predict.append((aspect, opinion, sentiment))
            
#             # Get the raw text for aspect and opinion
#             # Ambil token untuk aspect dan opinion
#             aspect_tokens = tokens[a:b+1]
#             opinion_tokens = tokens[c:d+1]
            
#             # Gabungkan subword jadi kata utuh
#             aspect_words = merge_subwords(aspect_tokens)
#             opinion_words = merge_subwords(opinion_tokens)
            
#             # Untuk aspect, ambil hanya kata pertama
#             aspect = aspect_words[0] if aspect_words else ""
            
#             # Untuk opinion, gabungkan semua kata
#             opinion = " ".join(opinion_words)
            
#             # Hilangkan tanda baca di akhir
#             opinion = opinion.rstrip('.,!?;:')
            
#             # Skip jika kosong
#             if not aspect or not opinion:
#                 continue
            
#             sentiment = RelationLabel(relations_probability[idx].argmax(-1).item()).name
            
#             if sentiment != RelationLabel.INVALID.name:
#                 predict.append((aspect, opinion, sentiment))

                
#         print("Text:", text)
#         print("Predict:", predict)
        
#         labels = []
#         words = text.split(" ")
#         for l in eval(label):
#             a, o, sm = l
#             a = " ".join([words[i] for i in a])
#             o = " ".join([words[i] for i in o])
#             labels.append((a, o, sm))
#         print("Label:", labels)
#         print("-" * 80)
        
#         res.append({"text": text, "predict": predict, "label": labels})

#     # Set default output path if not provided
#     if output_path is None:
#         # Extract folder name dari test_path untuk nama file
#         if os.path.isdir(test_path):
#             folder_name = os.path.basename(os.path.normpath(test_path))
#         else:
#             folder_name = os.path.basename(os.path.dirname(test_path))
#         output_path = f"test_results_{folder_name}.xlsx"
    
#     print(f"Saving results to {output_path}")
#     dataframe = pd.DataFrame(res)
#     dataframe.to_excel(output_path)
#     print("Testing completed successfully!")

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