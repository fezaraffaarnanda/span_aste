#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Debugging dan Analisis Mendalam Model SPAN-ASTE
===============================================
Kode ini bertujuan untuk visualisasi dan pemahaman detail setiap langkah 
dari proses fine-tuning dan inferensi model SPAN-ASTE dengan pendekatan teknis.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from typing import List, Tuple, Dict, Any, Optional

from models.model import SpanAsteModel
from models.metrics import SpanEvaluator, metrics
from models.losses import log_likelihood
from utils.tager import SpanLabel, RelationLabel, SentimentTriple, SentenceTagger

# Konstantisasi parameter penting untuk eksperimen
MAX_SEQ_LENGTH = 128
SPAN_WIDTH_EMBEDDING_DIM = 20
SPAN_MAXIMUM_LENGTH = 8
PRUNED_THRESHOLD = 0.5
PAIR_DISTANCE_EMBEDDING_DIM = 128
FFNN_HIDDEN_DIM = 150
LR_BERT = 5e-5
LR_OTHER = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.1
NUM_EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_CLIP = 1.0
SEED = 42

# Konfigurasi tampilan debug
VERBOSE_TOKENIZATION = True    # Tampilkan detail tokenisasi
VERBOSE_SPAN_REPR = True       # Tampilkan detail representasi span
VERBOSE_PAIR_FORMATION = True  # Tampilkan detail pembentukan pasangan
VERBOSE_FORWARD_PASS = True    # Tampilkan detail forward pass
VERBOSE_LOSS = True            # Tampilkan detail perhitungan loss
VERBOSE_GRADIENTS = False      # Tampilkan detail gradien (bisa sangat panjang)

def set_seed(seed=SEED):
    """
    Mengatur seed untuk reproduksibilitas
    """
    print(f"🔧 Menetapkan random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Konfigurasi tambahan untuk determinisme
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("✅ Random seed berhasil ditetapkan")

# Menggunakan GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Menggunakan device: {device}")

class ASPEDebugInfo:
    """
    Kelas untuk menyimpan dan menampilkan informasi debug
    """
    def __init__(self):
        self.tokenization_info = {}
        self.span_info = {}
        self.pair_info = {}
        self.forward_info = {}
        self.loss_info = {}
        self.gradient_info = {}
        
    def reset(self):
        """Reset semua informasi"""
        self.__init__()

# Instansiasi objek debug info
debug_info = ASPEDebugInfo()

class TripletsDataset(Dataset):
    """
    Dataset kustom untuk data triplet sentimen
    """
    def __init__(self, data: List[str], tokenizer, max_length=MAX_SEQ_LENGTH):
        """
        Args:
            data: List dari string dengan format "teks####[([target_indices], [opinion_indices], sentiment)]"
            tokenizer: BERT tokenizer
            max_length: Panjang maksimum sekuens
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = []  # Untuk menyimpan data yang sudah diproses
        
        print(f"🔍 Dataset dibuat dengan {len(data)} sampel")
        
        # Pre-process data untuk debugging
        for i, sample in enumerate(data[:5]):  # Limit 5 untuk debugging
            self._preprocess_sample(i, sample)
            
    def _preprocess_sample(self, idx, sample_str):
        """Pre-process sampel untuk debugging"""
        print(f"\n📊 PRE-PROCESSING SAMPEL #{idx+1}")
        print(f"📝 Raw input: {sample_str}")
        
        # Memisahkan teks dan anotasi
        parts = sample_str.split("####")
        text = parts[0]
        triplets_str = parts[1]
        
        # Parse triplets
        triplets = eval(triplets_str)
        print(f"🔍 Triplet yang diekstrak: {triplets}")
        
        # Tokenisasi text (untuk inspeksi)
        tokens = text.split()
        print(f"✂️ Token teks: {tokens}")
        
        # Token mapping untuk debugging
        debug_tokens = []
        bert_tokens = self.tokenizer.tokenize(text)
        print(f"🔡 BERT tokens: {bert_tokens}")
        
        offsets = []
        token_idx = 0
        
        # Map token asli ke BERT token untuk debugging
        for word_idx, word in enumerate(tokens):
            word_tokens = self.tokenizer.tokenize(word)
            start_idx = token_idx
            end_idx = token_idx + len(word_tokens) - 1
            
            debug_tokens.append({
                "word": word,
                "word_idx": word_idx,
                "bert_tokens": word_tokens,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            
            offsets.append((start_idx, end_idx))
            token_idx += len(word_tokens)
        
        if VERBOSE_TOKENIZATION:
            print("\n🔍 DETAIL MAPPING TOKEN:")
            for t in debug_tokens:
                print(f"  📌 Kata: '{t['word']}' (idx {t['word_idx']}) → BERT: {t['bert_tokens']} (idx {t['start_idx']}-{t['end_idx']})")
        
        # Simpan informasi token untuk debugging
        debug_info.tokenization_info[idx] = {
            "text": text,
            "tokens": tokens,
            "bert_tokens": bert_tokens,
            "mapping": debug_tokens,
            "offsets": offsets
        }
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Mengambil item dataset pada indeks tertentu
        """
        # Format data input: text####[([target_indices], [opinion_indices], sentiment)]
        sample_str = self.data[idx]
        
        # Memisahkan teks dan anotasi
        parts = sample_str.split("####")
        text = parts[0]
        triplets_str = parts[1]
        
        # Parse triplets
        triplets = eval(triplets_str)
        
        # Tokenize text
        tokens = text.split()
        
        # BERT tokenization
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        token_type_ids = inputs.token_type_ids
        
        # Hitung offset token BERT
        token_mapping = []
        bert_idx = 1  # Start after [CLS]
        
        for word_idx, word in enumerate(tokens):
            word_tokens = self.tokenizer.tokenize(word)
            start_idx = bert_idx
            end_idx = bert_idx + len(word_tokens) - 1
            token_mapping.append((start_idx, end_idx))
            bert_idx += len(word_tokens)
        
        # Konversi triplets ke format model
        sentiment_triples = []
        for triplet in triplets:
            aspect_indices = triplet[0]
            opinion_indices = triplet[1]
            sentiment = triplet[2]
            
            # Map indeks kata ke indeks token BERT
            if len(aspect_indices) > 0:
                a_start = token_mapping[aspect_indices[0]][0]
                a_end = token_mapping[aspect_indices[-1]][1]
            else:
                a_start, a_end = 0, 0
                
            if len(opinion_indices) > 0:
                o_start = token_mapping[opinion_indices[0]][0]
                o_end = token_mapping[opinion_indices[-1]][1]
            else:
                o_start, o_end = 0, 0
            
            # Konversi sentimen
            relation = {"POS": "POS", "NEG": "NEG", "NEU": "NEU"}
            sentiment_value = relation.get(sentiment, sentiment)
            
            triple = SentimentTriple(
                aspect=[a_start, a_end],
                opinion=[o_start, o_end],
                sentiment=sentiment_value
            )
            sentiment_triples.append(triple)
        
        # Buat SentenceTagger object
        sentence_tagger = SentenceTagger(sentiment_triples)
        spans, span_labels = sentence_tagger.spans
        relations, relation_labels = sentence_tagger.relations
        
        seq_len = len([i for i in input_ids if i != 0])
        
        return input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len


def visualize_span_detection(model, tokenizer, text, device=device):
    """
    Visualisasikan deteksi span untuk sebuah teks
    """
    print(f"\n📊 VISUALISASI DETEKSI SPAN")
    print(f"📝 Teks: \"{text}\"")
    
    # Tokenisasi
    inputs = tokenizer(
        text, 
        max_length=MAX_SEQ_LENGTH, 
        padding='max_length', 
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    token_type_ids = inputs.token_type_ids.to(device)
    seq_len = attention_mask.sum().item()
    
    # Forward pass untuk klasifikasi span
    with torch.no_grad():
        spans_probability, span_indices, _, _ = model(
            input_ids, attention_mask, token_type_ids, [seq_len]
        )
    
    # Decode span_indices ke teks
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Tampilkan top spans untuk Aspect dan Opinion
    print("\n🔍 TOP SPAN ASPECTS:")
    aspect_scores, aspect_indices = torch.topk(spans_probability[0, :, SpanLabel.ASPECT.value], k=5)
    for i, (score, idx) in enumerate(zip(aspect_scores, aspect_indices)):
        span_idx = span_indices[0][idx.item()]
        start, end = span_idx
        span_text = tokenizer.convert_tokens_to_string(tokens[start:end+1])
        print(f"  {i+1}. [{start},{end}] '{span_text}' (skor: {score.item():.4f})")
    
    print("\n🔍 TOP SPAN OPINIONS:")
    opinion_scores, opinion_indices = torch.topk(spans_probability[0, :, SpanLabel.OPINION.value], k=5)
    for i, (score, idx) in enumerate(zip(opinion_scores, opinion_indices)):
        span_idx = span_indices[0][idx.item()]
        start, end = span_idx
        span_text = tokenizer.convert_tokens_to_string(tokens[start:end+1])
        print(f"  {i+1}. [{start},{end}] '{span_text}' (skor: {score.item():.4f})")
    
    return spans_probability, span_indices


def visualize_relation_classification(model, tokenizer, text, device=device):
    """
    Visualisasikan klasifikasi relasi untuk sebuah teks
    """
    print(f"\n📊 VISUALISASI KLASIFIKASI RELASI")
    print(f"📝 Teks: \"{text}\"")
    
    # Tokenisasi
    inputs = tokenizer(
        text, 
        max_length=MAX_SEQ_LENGTH, 
        padding='max_length', 
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    token_type_ids = inputs.token_type_ids.to(device)
    seq_len = attention_mask.sum().item()
    
    # Forward pass
    with torch.no_grad():
        spans_probability, span_indices, relations_probability, candidate_indices = model(
            input_ids, attention_mask, token_type_ids, [seq_len]
        )
    
    # Decode token indices ke teks
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Tampilkan top relasi
    print("\n🔍 TOP RELATIONS:")
    for i in range(min(5, relations_probability.size(1))):
        relation_scores = relations_probability[0, i]
        best_rel_idx = torch.argmax(relation_scores).item()
        best_rel_score = relation_scores[best_rel_idx].item()
        
        if best_rel_idx != 0:  # Skip INVALID
            # Ambil indeks target dan opinion
            if i < len(candidate_indices[0]):
                a_start, a_end, o_start, o_end = candidate_indices[0][i]
                aspect_text = tokenizer.convert_tokens_to_string(tokens[a_start:a_end+1])
                opinion_text = tokenizer.convert_tokens_to_string(tokens[o_start:o_end+1])
                rel_name = RelationLabel(best_rel_idx).name
                print(f"  {i+1}. Aspek: '{aspect_text}' - Opini: '{opinion_text}' - Relasi: {rel_name} (skor: {best_rel_score:.4f})")
    
    return relations_probability, candidate_indices


def collate_fn(batch):
    """
    Fungsi kolasi untuk batching data
    """
    input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = zip(*batch)
    return input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len


def gold_labels(span_indices, spans, span_labels):
    """
    Memetakan indeks span prediksi ke label gold
    """
    if VERBOSE_FORWARD_PASS:
        print("\n📌 Menentukan Gold Labels:")
        
    gold_indices, gold_labels = [], []
    for batch_idx, indices in enumerate(span_indices):
        gold_ind, gold_lab = [], []
        
        if VERBOSE_FORWARD_PASS:
            print(f"  🔍 Batch {batch_idx} - Memproses {len(indices)} indeks")
            
        for i, indice in enumerate(indices):
            if indice in spans[batch_idx]:
                ix = spans[batch_idx].index(indice)
                gold_lab.append(span_labels[batch_idx][ix])
                
                if VERBOSE_FORWARD_PASS and i < 10:  # Batasi output untuk kecepatan
                    print(f"    ✅ Span {indice} cocok dengan label {span_labels[batch_idx][ix]}")
            else:
                gold_lab.append(0)
                
                if VERBOSE_FORWARD_PASS and i < 10:
                    print(f"    ❌ Span {indice} tidak cocok, diberi label 0")
                    
            gold_ind.append(indice)
        gold_indices.append(gold_ind)
        gold_labels.append(gold_lab)
    
    if VERBOSE_FORWARD_PASS:
        print(f"  ✅ Created {len(gold_indices)} gold indices groups")
        
    return gold_indices, gold_labels


def inspect_model_parameters(model):
    """
    Inspeksi parameter model secara detail
    """
    print("\n📊 INSPEKSI PARAMETER MODEL:")
    
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # Table header
    print(f"{'Nama Parameter':<40} {'Shape':<20} {'# Parameter':<15} {'Trainable':<10}")
    print("-" * 85)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            trainable_str = "✓"
        else:
            non_trainable_params += num_params
            trainable_str = "✗"
            
        shape_str = str(tuple(param.shape))
        print(f"{name:<40} {shape_str:<20} {num_params:<15,d} {trainable_str:<10}")
    
    # Summary
    print("-" * 85)
    print(f"Total Parameter: {total_params:,d}")
    print(f"Trainable Parameter: {trainable_params:,d}")
    print(f"Non-Trainable Parameter: {non_trainable_params:,d}")


def debug_span_representation(model, tokenizer, text, device=device):
    """
    Debug representasi span secara detail
    """
    print(f"\n🔍 DEBUG REPRESENTASI SPAN")
    print(f"📝 Teks: \"{text}\"")
    
    # Tokenisasi
    inputs = tokenizer(
        text, 
        max_length=MAX_SEQ_LENGTH, 
        padding='max_length', 
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    token_type_ids = inputs.token_type_ids.to(device)
    seq_len = attention_mask.sum().item()
    
    # Tokenisasi detail
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"\n📋 Tokens: {tokens[:seq_len]}")
    
    # Ambil output BERT saja
    with torch.no_grad():
        bert_output = model.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = bert_output.last_hidden_state
        
        # Generate spans (tanpa prediksi)
        spans, span_indices = model.span_representation(last_hidden_state, seq_len)
    
    # Tampilkan beberapa span
    print(f"\n📊 DETAIL SPAN:")
    max_display = min(10, len(span_indices))
    for i in range(max_display):
        span_idx = span_indices[i]
        start, end = span_idx
        span_text = tokenizer.convert_tokens_to_string(tokens[start:end+1])
        span_repr = spans[0, i]  # Batch 0
        
        # Calculate span width feature
        width = end - start + 1
        
        print(f"\n🔶 Span #{i}: [{start},{end}] '{span_text}' (width: {width})")
        print(f"   Token Awal: {tokens[start]} - Hidden State: {last_hidden_state[0, start][:3].cpu().numpy()}...")
        print(f"   Token Akhir: {tokens[end]} - Hidden State: {last_hidden_state[0, end][:3].cpu().numpy()}...")
        print(f"   Representasi Span: {span_repr[:5].cpu().numpy()}...")
    
    return spans, span_indices


def debug_forward_pass(model, tokenizer, text, device=device):
    """
    Debug forward pass secara detail
    """
    print(f"\n🔍 DEBUG FORWARD PASS")
    print(f"📝 Teks: \"{text}\"")
    
    # Tokenisasi
    inputs = tokenizer(
        text, 
        max_length=MAX_SEQ_LENGTH, 
        padding='max_length', 
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    token_type_ids = inputs.token_type_ids.to(device)
    seq_len = attention_mask.sum().item()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"\n📋 Tokens: {tokens[:seq_len]}")
    
    with torch.no_grad():
        # 1. Output BERT
        print("\n🔄 1. Encoding dengan BERT...")
        bert_output = model.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = bert_output.last_hidden_state
        print(f"   Shape: {last_hidden_state.shape}")
        print(f"   Sample values: {last_hidden_state[0, 1, :5].cpu().numpy()}...")
        
        # 2. Span Representation
        print("\n🔄 2. Membuat representasi span...")
        spans, span_indices = model.span_representation(last_hidden_state, seq_len)
        print(f"   Jumlah span: {len(span_indices)}")
        print(f"   Shape spans: {spans.shape}")
        
        # 3. Klasifikasi Span
        print("\n🔄 3. Klasifikasi span...")
        spans_probability = model.span_ffnn(spans)
        print(f"   Shape probability: {spans_probability.shape}")
        
        # 4. Pruning Span
        print("\n🔄 4. Pruning span...")
        nz = int(seq_len * model.span_pruned_threshold)
        print(f"   Jumlah span yang dipertahankan: {nz}")
        target_indices, opinion_indices = model.pruned_target_opinion(spans_probability, nz)
        
        # 5. Pembentukan Pasangan
        print("\n🔄 5. Pembentukan pasangan target-opinion...")
        candidates, candidate_indices, relation_indices = model.target_opinion_pair_representation(
            spans, span_indices, target_indices, opinion_indices)
        print(f"   Jumlah kandidat pasangan: {candidates.shape[1] if candidates.size(0) > 0 else 0}")
        
        # 6. Klasifikasi Relasi
        print("\n🔄 6. Klasifikasi relasi...")
        relations_probability = model.pairs_ffnn(candidates)
        print(f"   Shape probability relasi: {relations_probability.shape}")
        
        # 7. Tampilkan beberapa prediksi terbaik
        print("\n🔄 7. Top prediksi:")
        
        # Top spans untuk Aspect
        print("\n   🔍 TOP ASPECT SPANS:")
        aspect_scores, aspect_indices = torch.topk(spans_probability[0, :, SpanLabel.ASPECT.value], k=3)
        for i, (score, idx) in enumerate(zip(aspect_scores, aspect_indices)):
            span_idx = span_indices[idx.item()]
            start, end = span_idx
            span_text = tokenizer.convert_tokens_to_string(tokens[start:end+1])
            print(f"     {i+1}. [{start},{end}] '{span_text}' (skor: {score.item():.4f})")
        
        # Top spans untuk Opinion
        print("\n   🔍 TOP OPINION SPANS:")
        opinion_scores, opinion_indices = torch.topk(spans_probability[0, :, SpanLabel.OPINION.value], k=3)
        for i, (score, idx) in enumerate(zip(opinion_scores, opinion_indices)):
            span_idx = span_indices[idx.item()]
            start, end = span_idx
            span_text = tokenizer.convert_tokens_to_string(tokens[start:end+1])
            print(f"     {i+1}. [{start},{end}] '{span_text}' (skor: {score.item():.4f})")
        
        # Top relasi
        if relations_probability.size(1) > 0:
            print("\n   🔍 TOP RELATIONS:")
            for i in range(min(3, relations_probability.size(1))):
                relation_scores = relations_probability[0, i]
                best_rel_idx = torch.argmax(relation_scores).item()
                best_rel_score = relation_scores[best_rel_idx].item()
                
                if best_rel_idx != 0:  # Skip INVALID
                    # Ambil indeks target dan opinion
                    if i < len(candidate_indices[0]):
                        a_start, a_end, o_start, o_end = candidate_indices[0][i]
                        aspect_text = tokenizer.convert_tokens_to_string(tokens[a_start:a_end+1])
                        opinion_text = tokenizer.convert_tokens_to_string(tokens[o_start:o_end+1])
                        rel_name = RelationLabel(best_rel_idx).name
                        print(f"     {i+1}. Aspek: '{aspect_text}' - Opini: '{opinion_text}' - Relasi: {rel_name} (skor: {best_rel_score:.4f})")
    
    return spans_probability, span_indices, relations_probability, candidate_indices


def debug_training_step(model, optimizer, scheduler, batch, device=device):
    """
    Debug satu langkah training secara detail
    """
    print(f"\n🔄 DEBUG STEP TRAINING")
    
    # Unpack batch
    input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = batch
    
    # Konversi ke tensor
    input_ids = torch.tensor(input_ids, device=device)
    attention_mask = torch.tensor(attention_mask, device=device)
    token_type_ids = torch.tensor(token_type_ids, device=device)
    
    print(f"📝 Batch info - Size: {len(input_ids)}, Max seq length: {max(seq_len)}")
    
    # Forward pass
    print("\n🔄 1. Forward Pass...")
    spans_probability, span_indices, relations_probability, candidate_indices = model(
        input_ids, attention_mask, token_type_ids, seq_len)
    
    print(f"   Spans probability shape: {spans_probability.shape}")
    print(f"   Relations probability shape: {relations_probability.shape}")
    
    # Gold labels
    print("\n🔄 2. Menentukan Gold Labels...")
    gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
    gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
    
    # Loss calculation
    print("\n🔄 3. Perhitungan Loss...")
    loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)
    loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices, gold_relation_labels)
    
    # Total loss
    loss = 0.2 * loss_ner + loss_relation
    print(f"   Loss NER: {loss_ner.item():.4f}")
    print(f"   Loss Relation: {loss_relation.item():.4f}")
    print(f"   Total Loss: {loss.item():.4f}")
    
    # Backward
    print("\n🔄 4. Backward Pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # Print gradients jika diperlukan
    if VERBOSE_GRADIENTS:
        print("\n   🔍 Gradients:")
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"   {name}: {param.grad.norm().item():.6f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    print(f"   Gradient clipping at: {GRADIENT_CLIP}")
    
    # Update
    print("\n🔄 5. Parameter Update...")
    optimizer.step()
    scheduler.step()
    print(f"   Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    return loss.item()


def evaluate(model, metric, data_loader, tokenizer, device=device):
    """
    Evaluasi model pada dataset
    """
    print("\n🔍 EVALUASI MODEL")
    model.eval()
    metric.reset()
    
    all_predictions = []
    all_gold_triplets = []
    
    with torch.no_grad():
        for batch_ix, batch in enumerate(data_loader):
            print(f"\n⏳ Evaluasi batch {batch_ix+1}/{len(data_loader)}")
            input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = batch
            
            # Untuk analisis error
            raw_input_ids = input_ids
            
            # Konversi ke tensor
            input_ids = torch.tensor(input_ids, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            token_type_ids = torch.tensor(token_type_ids, device=device)
            
            print(f"📝 Batch info - Size: {len(input_ids)}, Max seq length: {max(seq_len)}")
            
            # Forward pass
            spans_probability, span_indices, relations_probability, candidate_indices = model(
                input_ids, attention_mask, token_type_ids, seq_len)
            
            print(f"✅ Forward pass selesai - Spans: {spans_probability.shape}, Relations: {relations_probability.shape}")
            
# Gold labels
            gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
            gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
            
            # Compute metrics
            num_correct, num_infer, num_label = metric.compute(relations_probability.cpu(),
                                                               torch.tensor(gold_relation_labels))
            
            print(f"📊 Metrics - Correct: {num_correct}, Inferred: {num_infer}, Label: {num_label}")
            metric.update(num_correct, num_infer, num_label)
            
            # Decode predictions untuk analisis
            predictions = []
            for batch_idx in range(relations_probability.size(0)):
                batch_preds = []
                tokens = tokenizer.convert_ids_to_tokens(raw_input_ids[batch_idx])
                
                for i in range(relations_probability.size(1)):
                    if i < len(candidate_indices[batch_idx]):
                        rel_scores = relations_probability[batch_idx, i]
                        rel_idx = torch.argmax(rel_scores).item()
                        
                        if rel_idx != 0:  # Skip INVALID
                            a_start, a_end, o_start, o_end = candidate_indices[batch_idx][i]
                            aspect_text = tokenizer.convert_tokens_to_string(tokens[a_start:a_end+1])
                            opinion_text = tokenizer.convert_tokens_to_string(tokens[o_start:o_end+1])
                            rel_name = RelationLabel(rel_idx).name
                            
                            batch_preds.append((aspect_text, opinion_text, rel_name))
                
                predictions.append(batch_preds)
                
                # Print beberapa prediksi
                if batch_idx < 2:  # Limit output
                    print(f"\n🔍 Prediksi untuk sampel {batch_idx+1}:")
                    for j, pred in enumerate(batch_preds[:3]):  # Show top 3
                        print(f"  {j+1}. {pred}")
            
            all_predictions.extend(predictions)
            
            # Extract gold triplets
            gold_triplets = []
            for batch_idx in range(len(raw_input_ids)):
                tokens = tokenizer.convert_ids_to_tokens(raw_input_ids[batch_idx])
                batch_triplets = []
                
                # Extract from relations
                for rel_idx, rel in enumerate(relations[batch_idx]):
                    if rel_idx < len(relation_labels[batch_idx]):
                        label = relation_labels[batch_idx][rel_idx]
                        if label != 0:  # Skip INVALID
                            a_start, a_end, o_start, o_end = rel
                            aspect_text = tokenizer.convert_tokens_to_string(tokens[a_start:a_end+1])
                            opinion_text = tokenizer.convert_tokens_to_string(tokens[o_start:o_end+1])
                            rel_name = RelationLabel(label).name
                            
                            batch_triplets.append((aspect_text, opinion_text, rel_name))
                
                gold_triplets.append(batch_triplets)
            
            all_gold_triplets.extend(gold_triplets)
    
    precision, recall, f1 = metric.accumulate()
    print(f"\n📊 Hasil Evaluasi:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    # Analisis error (tampilkan beberapa contoh error)
    print("\n🔍 ANALISIS ERROR:")
    for i in range(min(5, len(all_predictions))):
        if all_predictions[i] != all_gold_triplets[i]:
            print(f"\n📝 Sampel #{i+1}:")
            print(f"  Gold: {all_gold_triplets[i]}")
            print(f"  Pred: {all_predictions[i]}")
            
            # Analisis jenis error
            gold_set = set(all_gold_triplets[i])
            pred_set = set(all_predictions[i])
            
            false_positives = pred_set - gold_set
            false_negatives = gold_set - pred_set
            
            if false_positives:
                print(f"  False Positives: {false_positives}")
            if false_negatives:
                print(f"  False Negatives: {false_negatives}")
    
    model.train()
    return precision, recall, f1


def visualize_training(epochs, train_losses, val_metrics):
    """
    Visualisasi hasil training
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot validation metrics
    plt.subplot(1, 3, 2)
    val_epochs = list(range(1, len(val_metrics[0])+1))
    plt.plot(val_epochs, val_metrics[0], 'r-o', label='Precision')
    plt.plot(val_epochs, val_metrics[1], 'g-o', label='Recall')
    plt.plot(val_epochs, val_metrics[2], 'b-o', label='F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 trend
    plt.subplot(1, 3, 3)
    plt.plot(val_epochs, val_metrics[2], 'b-o')
    plt.title('F1 Score Trend')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_visualization.png')
    print("\n📊 Visualisasi training disimpan sebagai 'training_visualization.png'")


# Main Program
if __name__ == "__main__":
    # Set seed untuk reproduksibilitas
    set_seed(SEED)
    
    # Contoh data
    example_data = [
        "apaan sih baru di download nggak bisa dibuka.####[([4], [5, 6, 7], 'NEG')]",
        "kalo ga bisa bikin aplikasi ga ush bikin nyusahin doang.####[([4], [1, 2, 3], 'NEG')]",
        "tolong google berikan opsi untuk 0 bintang , aplikasi asu.####[([8], [9], 'NEG')]",
        "scannya lamaaa.####[([0], [1], 'NEG')]",
        "aplikasi yang sangat bermanfaat bagi masyarakat kota yogyskarta.####[([0], [2, 3], 'POS')]",
        "looodiiiing terussss . mending gausah bikin app daripada bikin kesal orang.####[([0], [1], 'NEG'), ([6], [8, 9, 10], 'NEG')]",
        "kalau siang buka apl nya loading agak lambat.####[([3], [6, 7], 'NEG')]",
        "setiap akan pinjam buku selalu ada peringatan error dan kadang tiba2 ada peringatan aplikasi terhenti.####[([13], [14], 'NEG'), ([2, 3], [7], 'NEG')]",
        "aplikasi tidak bisa dibuka.####[([0], [1, 2, 3], 'NEG')]",
        "sangatt membantu , jadinya ga perlu ribet bolak - balik . adminnya fast respon jadi kalau ada kendala langsung teratasi . aplikasinya mantap banget.####[([11], [12, 13], 'POS'), ([21], [22, 23], 'POS')]"
    ]
    
    # Inisialisasi tokenizer BERT
    print("\n🔄 Menginisialisasi BERT Tokenizer")
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
    print(f"✅ Tokenizer berhasil diinisialisasi, ukuran vocab: {len(tokenizer.vocab)}")
    
    # Persiapkan dataset
    print("\n🔄 Menyiapkan Dataset")
    dataset = TripletsDataset(example_data, tokenizer, max_length=MAX_SEQ_LENGTH)
    
    # Split dataset untuk validasi
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"✅ Dataset dibagi menjadi {train_size} sampel training dan {val_size} sampel validasi")
    
    # Inisialisasi model
    print("\n🔄 Menginisialisasi Model SPAN-ASTE")
    target_dim = len(SpanLabel)
    relation_dim = len(RelationLabel)
    print(f"📊 Dimensi target: {target_dim}, Dimensi relasi: {relation_dim}")
    
    model = SpanAsteModel(
        'indobenchmark/indobert-base-p2',
        target_dim,
        relation_dim,
        ffnn_hidden_dim=FFNN_HIDDEN_DIM,
        span_width_embedding_dim=SPAN_WIDTH_EMBEDDING_DIM,
        span_maximum_length=SPAN_MAXIMUM_LENGTH,
        span_pruned_threshold=PRUNED_THRESHOLD,
        pair_distance_embeddings_dim=PAIR_DISTANCE_EMBEDDING_DIM,
        device=device
    )
    model.to(device)
    
    # Inspeksi detail model
    inspect_model_parameters(model)
    
    # Optimizer dan scheduler
    print("\n🔄 Menyiapkan Optimizer dan Scheduler")
    no_decay = ['bias', 'LayerNorm.weight']
    bert_param_optimizer = list(model.bert.named_parameters())
    span_linear_param_optimizer = list(model.span_ffnn.named_parameters())
    pair_linear_param_optimizer = list(model.pairs_ffnn.named_parameters())
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY, 'lr': LR_BERT},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': LR_BERT},
        
        {'params': [p for n, p in span_linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': LR_OTHER},
        {'params': [p for n, p in span_linear_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': LR_OTHER},
        
        {'params': [p for n, p in pair_linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': LR_OTHER},
        {'params': [p for n, p in pair_linear_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': LR_OTHER}
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_PROPORTION)
    
    print(f"📊 Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Metrik evaluasi
    metric = SpanEvaluator()
    
    # Debug representasi span dengan contoh
    debug_text = "aplikasi sangat bermanfaat bagi masyarakat"
    debug_span_representation(model, tokenizer, debug_text)
    
    # Debug forward pass dengan contoh
    debug_forward_pass(model, tokenizer, debug_text)
    
    # Visualisasi deteksi span dengan contoh
    visualize_span_detection(model, tokenizer, debug_text)
    
    # Visualisasi klasifikasi relasi dengan contoh
    visualize_relation_classification(model, tokenizer, debug_text)
    
    # Training model
    print("\n🚀 MEMULAI TRAINING MODEL")
    global_step = 0
    best_f1 = 0.0
    train_losses = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n📆 EPOCH {epoch+1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            print(f"\n⏳ Batch {batch_idx+1}/{len(train_dataloader)}")
            
            # Debug training step
            loss = debug_training_step(model, optimizer, scheduler, batch)
            
            # Simpan loss
            epoch_loss += loss
            train_losses.append(loss)
            
            global_step += 1
            
            # Evaluasi setiap 2 batch (untuk contoh)
            if global_step % 2 == 0 or batch_idx == len(train_dataloader) - 1:
                print(f"\n🔍 Evaluasi model pada epoch {epoch+1}, step {global_step}")
                precision, recall, f1 = evaluate(model, metric, val_dataloader, tokenizer)
                
                if f1 > best_f1:
                    best_f1 = f1
                    print(f"✅ F1 score baru terbaik: {best_f1:.4f}")
                    torch.save(model.state_dict(), "best_model.pt")
        
        # Evaluasi pada akhir epoch
        precision, recall, f1 = evaluate(model, metric, val_dataloader, tokenizer)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        
        print(f"\n📊 Ringkasan Epoch {epoch+1}:")
        print(f"  Loss rata-rata: {epoch_loss / len(train_dataloader):.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
    
    print("\n✅ TRAINING SELESAI")
    print(f"🏆 Best F1 score: {best_f1:.4f}")
    
    # Visualisasi hasil training
    visualize_training(NUM_EPOCHS, train_losses, [val_precisions, val_recalls, val_f1s])
    
    # Inferensi pada contoh baru
    print("\n🚀 INFERENSI PADA CONTOH BARU")
    
    # Muat model terbaik
    model.load_state_dict(torch.load("checkpoint/fold1/model_best/model.pt"))
    model.eval()
    
    # Contoh baru
    new_examples = [
        "bukannya mempermudah , malah mempersulit . setelah mendaftar tidak bisa langsung login malah suruh nunggu aktivasi dari pihak desa , permasalahannya butuh berapa minggu untuk menunggu aktivasi dari desa ? katanya ngurus apa suruh online , giliran udah daftar online , eh masih aja nunggu .",
        "aplikasi ga guna sampah , nama aplikasi sebelumnya peduli lindungi . udah pernah daftar di suruh daftar lagi.",
        "idenya bagus tapi sayang lemot banget."
    ]
    
    for i, text in enumerate(new_examples):
        print(f"\n📝 Contoh #{i+1}: \"{text}\"")
        
        # Debug forward pass
        spans_probability, span_indices, relations_probability, candidate_indices = debug_forward_pass(model, tokenizer, text)
        
        # Tampilkan hasil inferensi
        print("\n🔍 HASIL INFERENSI:")
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Extract triplets
        triplets = []
        for i in range(min(5, relations_probability.size(1))):
            if i < len(candidate_indices[0]):
                rel_scores = relations_probability[0, i]
                rel_idx = torch.argmax(rel_scores).item()
                rel_score = rel_scores[rel_idx].item()
                
                if rel_idx != 0 and rel_score > 0.5:  # Skip INVALID & low confidence
                    a_start, a_end, o_start, o_end = candidate_indices[0][i]
                    aspect_text = tokenizer.convert_tokens_to_string(decoded_tokens[a_start:a_end+1])
                    opinion_text = tokenizer.convert_tokens_to_string(decoded_tokens[o_start:o_end+1])
                    rel_name = RelationLabel(rel_idx).name
                    
                    triplets.append((aspect_text, opinion_text, rel_name))
        
        print(f"  Triplet terdeteksi: {triplets}")
        
        # Analisis detail (aspect, opinion terdeteksi)
        print("\n📊 DETAIL ANALISIS:")
        print("  Aspek terdeteksi:")
        aspect_scores, aspect_indices = torch.topk(spans_probability[0, :, SpanLabel.ASPECT.value], k=5)
        for i, (score, idx) in enumerate(zip(aspect_scores, aspect_indices)):
            span_idx = span_indices[idx.item()]
            start, end = span_idx
            span_text = tokenizer.convert_tokens_to_string(decoded_tokens[start:end+1])
            print(f"    {i+1}. '{span_text}' (skor: {score.item():.4f})")
        
        print("\n  Opinion terdeteksi:")
        opinion_scores, opinion_indices = torch.topk(spans_probability[0, :, SpanLabel.OPINION.value], k=5)
        for i, (score, idx) in enumerate(zip(opinion_scores, opinion_indices)):
            span_idx = span_indices[idx.item()]
            start, end = span_idx
            span_text = tokenizer.convert_tokens_to_string(decoded_tokens[start:end+1])
            print(f"    {i+1}. '{span_text}' (skor: {score.item():.4f})")
    
    print("\n✅ ANALISIS SELESAI")