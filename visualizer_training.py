#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SPAN-ASTE Model Visualizer
This script traces through the execution of Span-ASTE model to help understand:
- Input processing
- Model architecture and forward pass
- Fine-tuning process
- Output generation
"""

import os
import torch
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from pprint import pprint
import colorama
from colorama import Fore, Back, Style

# Import your modules
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.processor import Res15DataProcessor
from utils.tager import SpanLabel, RelationLabel
from models.collate import collate_fn
from models.losses import log_likelihood
from models.metrics import SpanEvaluator

# Initialize colorama
colorama.init()

def print_header(text):
    """Print a section header with formatting"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "="*80)
    print(f" {text}")
    print("=" * 80 + f"{Style.RESET_ALL}\n")

def print_subheader(text):
    """Print a subsection header with formatting"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}" + "-"*50)
    print(f" {text}")
    print("-" * 50 + f"{Style.RESET_ALL}\n")

def print_info(text, data=None):
    """Print info with formatting"""
    print(f"{Fore.YELLOW}➤ {text}:{Style.RESET_ALL}")
    if data is not None:
        if isinstance(data, torch.Tensor):
            print(f"  Shape: {data.shape}")
            print(f"  Data: {data}")
        else:
            print(f"  {data}")
    print()

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Custom gold_labels function to avoid errors
def custom_gold_labels(span_indices, spans, span_labels):
    """
    Organizing gold labels and indices - modified to handle visualization data
    """
    # Ensure spans and span_labels are lists with at least one element
    if not spans:
        spans = [[]]
    if not span_labels:
        span_labels = [[]]
    
    # gold span labels
    gold_indices, gold_labels = [], []
    for batch_idx, indices in enumerate(span_indices):
        gold_ind, gold_lab = [], []
        for indice in indices:
            # Check if batch_idx is valid for spans
            if batch_idx < len(spans):
                # Check if indice is in spans[batch_idx]
                if indice in spans[batch_idx]:
                    ix = spans[batch_idx].index(indice)
                    if ix < len(span_labels[batch_idx]):
                        gold_lab.append(span_labels[batch_idx][ix])
                    else:
                        gold_lab.append(0)
                else:
                    gold_lab.append(0)
            else:
                gold_lab.append(0)
            gold_ind.append(indice)
        gold_indices.append(gold_ind)
        gold_labels.append(gold_lab)

    return gold_indices, gold_labels

def visualize_model():
    """Trace through the SPAN-ASTE model execution for visualization"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--data_path", default="data/15res", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--span_maximum_length", default=8, type=int)
    parser.add_argument("--span_pruned_threshold", default=0.5, type=float)
    parser.add_argument("--save_dir", default='./visualization_output', type=str)
    parser.add_argument("--seed", default=1024, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print_header("SPAN-ASTE MODEL VISUALIZATION")
    print_info("Using device", device)
    
    # 1. Initialize tokenizer and processor
    print_header("1. INITIALIZATION")
    print_subheader("1.1 Tokenizer and Processor")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    print_info("Tokenizer initialized", tokenizer.__class__.__name__)
    
    processor = Res15DataProcessor(tokenizer, args.max_seq_len)
    print_info("Data processor initialized", processor.__class__.__name__)
    
    # 2. Load dataset
    print_subheader("1.2 Dataset Loading")
    print_info("Loading sample data from", args.data_path)
    
    # Create a fake example for visualization if data path doesn't exist
    if not os.path.exists(args.data_path):
        print_info("Data path doesn't exist, creating synthetic example")
        # Create a directory for our synthetic example
        os.makedirs(args.data_path, exist_ok=True)
        with open(os.path.join(args.data_path, "train_triplets.txt"), "w") as f:
            f.write("The pizza has a delicious taste and fantastic texture.####[([2], [5], 'POS'), ([2], [8], 'POS')]")
        with open(os.path.join(args.data_path, "dev_triplets.txt"), "w") as f:
            f.write("The pizza has a delicious taste and fantastic texture.####[([2], [5], 'POS'), ([2], [8], 'POS')]")
    
    try:
        train_dataset = CustomDataset("train", args.data_path, processor, tokenizer, args.max_seq_len)
        print_info("Train dataset loaded", f"Examples: {len(train_dataset)}")
        
        # 3. Setup dataloader
        print_subheader("1.3 DataLoader Construction")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        print_info("DataLoader constructed", f"Batches: {len(train_dataloader)}")
        
        # Get a single batch
        try:
            batch = next(iter(train_dataloader))
            input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = batch
        except (StopIteration, ValueError) as e:
            print_info(f"Error getting batch: {e}. Using synthetic data instead.")
            # Create synthetic batch data
            input_ids = [[101, 1996, 13971, 2003, 1037, 26462, 3899, 1998, 22496, 7328, 102, 0, 0]]
            attention_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
            token_type_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            spans = [[(2, 2), (5, 5), (8, 8)]]
            span_labels = [[1, 2, 2]]  # 1=Aspect, 2=Opinion
            relations = [[(2, 2, 5, 5), (2, 2, 8, 8)]]
            relation_labels = [[1, 1]]  # 1=POS
            seq_len = [11]  # Length including CLS and SEP tokens
    except Exception as e:
        print_info(f"Error loading dataset: {e}. Using synthetic data instead.")
        # Create synthetic batch data
        input_ids = [[101, 1996, 13971, 2003, 1037, 26462, 3899, 1998, 22496, 7328, 102, 0, 0]]
        attention_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        token_type_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        spans = [[(2, 2), (5, 5), (8, 8)]]
        span_labels = [[1, 2, 2]]  # 1=Aspect, 2=Opinion
        relations = [[(2, 2, 5, 5), (2, 2, 8, 8)]]
        relation_labels = [[1, 1]]  # 1=POS
        seq_len = [11]  # Length including CLS and SEP tokens
        
        # Create dataloader placeholder
        train_dataloader = [batch]
    
    # 4. Build model
    print_header("2. MODEL ARCHITECTURE")
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
    print_info("Target dimension", target_dim)
    print_info("Relation dimension", relation_dim)
    
    model = SpanAsteModel(
        args.bert_model,
        target_dim,
        relation_dim,
        span_maximum_length=args.span_maximum_length,
        span_pruned_threshold=args.span_pruned_threshold,
        device=device
    )
    model.to(device)
    print_info("Model constructed", model.__class__.__name__)
    
    # Print model architecture
    print_subheader("2.1 Model Components")
    print_info("Main components")
    for name, module in model.named_children():
        print(f"  - {name}: {module.__class__.__name__}")
    
    # 5. Optimizer setup
    print_header("3. TRAINING SETUP")
    print_subheader("3.1 Optimizer Configuration")
    
    no_decay = ['bias', 'LayerNorm.weight']
    bert_param_optimizer = list(model.bert.named_parameters())
    span_linear_param_optimizer = list(model.span_ffnn.named_parameters())
    pair_linear_param_optimizer = list(model.pairs_ffnn.named_parameters())
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 1e-2, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in span_linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-3},
        {'params': [p for n, p in span_linear_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': 1e-3},
        {'params': [p for n, p in pair_linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-3},
        {'params': [p for n, p in pair_linear_param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': 1e-3}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    print_info("Optimizer", optimizer.__class__.__name__)
    
    metric = SpanEvaluator()
    print_info("Evaluation metric", metric.__class__.__name__)
    
    # 6. Single batch processing visualization
    print_header("4. FORWARD PASS VISUALIZATION")
    model.train()
    
    print_subheader("4.1 Input Data")
    print_info("Input IDs shape", f"{len(input_ids)}×{len(input_ids[0])}")
    
    # Decode the first example to see the text
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    text = tokenizer.decode(input_ids[0])
    print_info("Decoded text", text)
    print_info("Tokens", tokens[:20] + ['...'] if len(tokens) > 20 else tokens)
    
    # Put tensors on device
    input_ids = torch.tensor(input_ids, device=device)
    attention_mask = torch.tensor(attention_mask, device=device)
    token_type_ids = torch.tensor(token_type_ids, device=device)
    
    print_info("Sequence length", seq_len)
    print_info("Spans (gold)", spans)
    print_info("Span labels (gold)", span_labels)
    print_info("Relations (gold)", relations)
    print_info("Relation labels (gold)", relation_labels)
    
    # 7. Forward pass
    print_subheader("4.2 Model Forward Pass Execution")
    print_info("Executing model forward pass...")
    
    with torch.no_grad():
        # Execute the full forward pass
        try:
            spans_probability, span_indices, relations_probability, candidate_indices = model(
                input_ids, attention_mask, token_type_ids, seq_len)
            forward_success = True
        except Exception as e:
            print_info(f"Error in forward pass: {e}")
            forward_success = False
            # Continue with step-by-step execution
        
        # Step 1: BERT encoding
        print_info("Step 1: BERT encoding")
        bert_output = model.bert(input_ids, attention_mask, token_type_ids)
        x = bert_output.last_hidden_state
        print_info("BERT output shape", x.shape)
        
        # Step 2: Span representation
        print_info("Step 2: Span representation")
        max_seq_len = max(seq_len)
        spans_tensor, span_indices = model.span_representation(x, max_seq_len)
        print_info("Spans tensor shape", spans_tensor.shape)
        print_info("Number of candidate spans", len(span_indices))
        print_info("Sample span indices", span_indices[:5])
        
        # Step 3: Span classification
        print_info("Step 3: Span classification (FFNN)")
        spans_probability = model.span_ffnn(spans_tensor)
        print_info("Spans probability shape", spans_probability.shape)
        print_info("Span probability sample", spans_probability[0, 0])
        
        # Step 4: Pruned target opinion
        print_info("Step 4: Pruning spans")
        nz = int(max_seq_len * model.span_pruned_threshold)
        print_info("Top-k spans to keep", nz)
        target_indices, opinion_indices = model.pruned_target_opinion(spans_probability, nz)
        print_info("Target indices shape", target_indices.shape)
        print_info("Opinion indices shape", opinion_indices.shape)
        
        # Step 5: Target-opinion pair representation
        print_info("Step 5: Target-opinion pair representation")
        try:
            candidates, candidate_indices, relation_indices = model.target_opinion_pair_representation(
                spans_tensor, span_indices, target_indices, opinion_indices)
            print_info("Candidates shape", candidates.shape)
            print_info("Number of candidate pairs", len(candidate_indices[0]))
            
            # Step 6: Relation classification
            print_info("Step 6: Relation classification (FFNN)")
            relations_probability = model.pairs_ffnn(candidates)
            print_info("Relations probability shape", relations_probability.shape)
            pair_success = True
        except Exception as e:
            print_info(f"Error in pair representation: {e}")
            pair_success = False
    
    # 8. Loss computation
    print_subheader("4.3 Loss Computation")
    
    # Get gold labels for spans using custom function
    try:
        gold_span_indices, gold_span_labels = custom_gold_labels(span_indices, spans, span_labels)
        print_info("Gold span indices (sample)", gold_span_indices[0][:5] if gold_span_indices and gold_span_indices[0] else "None")
        print_info("Gold span labels (sample)", gold_span_labels[0][:5] if gold_span_labels and gold_span_labels[0] else "None")
        
        # Calculate span loss
        if forward_success:
            loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)
            print_info("Span detection loss", loss_ner.item())
        else:
            print_info("Skipping span loss calculation due to forward pass error")
        
        # Get gold labels for relations
        if pair_success:
            gold_relation_indices, gold_relation_labels = custom_gold_labels(candidate_indices, relations, relation_labels)
            print_info("Gold relation indices (sample)", gold_relation_indices[0][:5] if gold_relation_indices and gold_relation_indices[0] else "None")
            print_info("Gold relation labels (sample)", gold_relation_labels[0][:5] if gold_relation_labels and gold_relation_labels[0] else "None")
            
            # Calculate relation loss
            if not relations or all(not rl for rl in relation_labels):
                print_info("Skipping relation loss calculation (no gold relations)")
                loss_relation = torch.tensor(0.0, device=device)
            else:
                loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices, gold_relation_labels)
                print_info("Relation extraction loss", loss_relation.item())
            
            # Combined loss
            if forward_success:
                loss = 0.2 * loss_ner + loss_relation
                print_info("Combined loss", loss.item())
        else:
            print_info("Skipping relation loss calculation due to pair representation error")
    except Exception as e:
        print_info(f"Error in loss computation: {e}")
    
    # 9. Backward pass & optimization (conceptual)
    print_subheader("4.4 Backward Pass (conceptual)")
    print_info("In training, we'd now:")
    print_info("1. Zero gradients", "optimizer.zero_grad()")
    print_info("2. Backward pass", "loss.backward()")
    print_info("3. Update weights", "optimizer.step()")
    
    # 10. Prediction process
    print_header("5. PREDICTION PROCESS")
    model.eval()
    
    print_subheader("5.1 Model Output Processing")
    # Forward pass for prediction
    if forward_success and pair_success:
        print_info("Final spans probability", spans_probability.shape)
        print_info("Final relations probability", relations_probability.shape)
        
        print_subheader("5.2 Extraction of Predictions")
        # Get predicted aspect and opinion spans
        aspect_predictions = []
        for batch_idx in range(spans_probability.size(0)):
            for span_idx in range(spans_probability.size(1)):
                if spans_probability[batch_idx, span_idx, SpanLabel.ASPECT.value].item() > 0.5:
                    span = span_indices[batch_idx][span_idx]
                    start, end = span
                    span_text = tokenizer.decode(input_ids[batch_idx][start:end+1])
                    aspect_predictions.append((start, end, span_text))
        
        opinion_predictions = []
        for batch_idx in range(spans_probability.size(0)):
            for span_idx in range(spans_probability.size(1)):
                if spans_probability[batch_idx, span_idx, SpanLabel.OPINION.value].item() > 0.5:
                    span = span_indices[batch_idx][span_idx]
                    start, end = span
                    span_text = tokenizer.decode(input_ids[batch_idx][start:end+1])
                    opinion_predictions.append((start, end, span_text))
        
        print_info("Predicted aspect spans", aspect_predictions)
        print_info("Predicted opinion spans", opinion_predictions)
        
        # Get predicted sentiment triplets
        triplet_predictions = []
        for batch_idx in range(relations_probability.size(0)):
            for triplet_idx in range(relations_probability.size(1)):
                # Get the most likely sentiment
                sentiment_id = relations_probability[batch_idx, triplet_idx].argmax(-1).item()
                if sentiment_id != RelationLabel.INVALID.value:
                    a, b, c, d = candidate_indices[batch_idx][triplet_idx]
                    aspect = tokenizer.decode(input_ids[batch_idx][a:b+1])
                    opinion = tokenizer.decode(input_ids[batch_idx][c:d+1])
                    sentiment = RelationLabel(sentiment_id).name
                    
                    triplet_predictions.append((aspect, opinion, sentiment))
        
        print_info("Predicted triplets", triplet_predictions)
        
        # 11. Evaluation metrics
        print_header("6. EVALUATION METRICS")
        metric = SpanEvaluator()
        
        # Prepare dummy gold labels for demonstration
        dummy_gold_labels = torch.zeros_like(relations_probability.argmax(-1))
        if relations and relation_labels and relation_labels[0]:
            for b_idx, batch_relations in enumerate(relations):
                for r_idx, relation in enumerate(batch_relations):
                    if relation in candidate_indices[b_idx]:
                        c_idx = candidate_indices[b_idx].index(relation)
                        dummy_gold_labels[b_idx, c_idx] = relation_labels[b_idx][r_idx]
        
        num_correct, num_infer, num_label = metric.compute(
            relations_probability, dummy_gold_labels)
        
        print_info("Metrics computation")
        print_info("Number of correct predictions", num_correct)
        print_info("Number of predictions", num_infer)
        print_info("Number of gold labels", num_label)
        
        # Update metrics
        metric.update(num_correct, num_infer, num_label)
        precision, recall, f1 = metric.accumulate()
        
        print_info("Evaluation results")
        print_info("Precision", precision)
        print_info("Recall", recall)
        print_info("F1 Score", f1)
    else:
        print_info("Skipping prediction visualization due to errors in forward pass")
    
    print_header("MODEL EXECUTION VISUALIZATION COMPLETE")
    print("This visualization has walked through the entire Span-ASTE model process, from input to evaluation.")
    print("Any errors encountered were handled to show as much of the process as possible.")

if __name__ == "__main__":
    visualize_model()