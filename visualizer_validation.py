#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Debugging script to visualize SPAN-ASTE model processing
This script demonstrates how the SPAN-ASTE model processes data from input to output
"""

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.collate import collate_fn, gold_labels
from models.model import SpanAsteModel
from utils.dataset import CustomDataset
from utils.processor import Res15DataProcessor
from utils.tager import SpanLabel, RelationLabel

def set_seed(seed=1024):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def debug_visualize_aste_model(model_path, data_path, bert_model, max_seq_len=128, batch_size=1):
    """
    Visualize the complete processing pipeline of SPAN-ASTE model
    
    Args:
        model_path: Path to saved model or directory containing model.pt
        data_path: Path to data directory containing dev_triplets.txt
        bert_model: Name or path of BERT model
        max_seq_len: Maximum sequence length for tokenizer
        batch_size: Batch size (should be 1 for clearer visualization)
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\n[1] Loading tokenizer: {bert_model}")
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    
    # Create data processor
    print(f"\n[2] Creating data processor with max_seq_len={max_seq_len}")
    processor = Res15DataProcessor(tokenizer, max_seq_len)
    
    # Load dataset (using dev set for visualization)
    print(f"\n[3] Loading dataset from {data_path}")
    dataset = CustomDataset("dev", data_path, processor, tokenizer, max_seq_len)
    print(f"  Dataset loaded with {len(dataset)} examples")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Initialize model
    print(f"\n[4] Building SPAN-ASTE model")
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
    print(f"  Target dimension (SpanLabel): {target_dim} - {[label.name for label in SpanLabel]}")
    print(f"  Relation dimension (RelationLabel): {relation_dim} - {[label.name for label in RelationLabel]}")
    
    model = SpanAsteModel(
        bert_model,
        target_dim,
        relation_dim,
        span_maximum_length=8,  # Maximum span length
        span_pruned_threshold=0.5,  # Threshold for span pruning
        device=device
    )
    
    # Load model weights
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pt")
    
    print(f"\n[5] Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    # Process a single batch for visualization
    print(f"\n[6] Processing example data to visualize model steps")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = batch
            
            # Print example information
            print("\n" + "="*80)
            print(f"EXAMPLE {batch_idx+1}:")
            print("="*80)
            
            # Convert input_ids to text for better visualization
            print("\n[6.1] Input Text:")
            for i in range(len(input_ids)):
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
                text = tokenizer.convert_tokens_to_string(tokens)
                print(f"  Example {i+1}: \"{text}\"")
                
                # Print gold spans and labels
                print("\n  Gold Spans (ground truth):")
                for j, (span, label) in enumerate(zip(spans[i], span_labels[i])):
                    span_text = tokenizer.convert_tokens_to_string(tokens[span[0]:span[1]+1])
                    span_type = SpanLabel(label).name
                    print(f"    Span {j+1}: {span} -> \"{span_text}\" (Type: {span_type})")
                
                # Print gold relations
                print("\n  Gold Relations (ground truth triplets):")
                for j, (relation, label) in enumerate(zip(relations[i], relation_labels[i])):
                    if len(relation) == 4:
                        a, b, c, d = relation
                        aspect = tokenizer.convert_tokens_to_string(tokens[a:b+1])
                        opinion = tokenizer.convert_tokens_to_string(tokens[c:d+1])
                        sentiment = RelationLabel(label).name
                        print(f"    Triplet {j+1}: ({a},{b},{c},{d}) -> Aspect: \"{aspect}\", Opinion: \"{opinion}\", Sentiment: {sentiment}")
            
            # Move tensors to device
            input_ids = torch.tensor(input_ids, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            token_type_ids = torch.tensor(token_type_ids, device=device)
            
            # STEP 1: Get BERT embeddings
            print("\n[6.2] BERT Encoder Processing:")
            bert_output = model.bert(input_ids, attention_mask, token_type_ids)
            last_hidden_state = bert_output.last_hidden_state
            print(f"  BERT output shape: {last_hidden_state.shape}")
            print(f"  This represents contextualized embeddings for each token in the input sequence")
            
            # STEP 2: Generate spans
            print("\n[6.3] Span Representation Generation:")
            spans, span_indices = model.span_representation(last_hidden_state, max(seq_len))
            print(f"  Generated {spans.shape[1]} possible spans with representation dimension {spans.shape[2]}")
            print(f"  Each span is represented as [hi; hj; f_width(i,j)] where:")
            print(f"    - hi is the BERT embedding of the start token")
            print(f"    - hj is the BERT embedding of the end token")
            print(f"    - f_width(i,j) is a learned embedding of the span width")
            
            # Print some example spans
            print("\n  Example spans:")
            for i in range(min(5, len(span_indices))):
                start, end = span_indices[i]
                span_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start:end+1])
                span_text = tokenizer.convert_tokens_to_string(span_tokens)
                print(f"    Span {i+1}: ({start},{end}) -> \"{span_text}\" (width: {end-start+1})")
            print(f"    ... (total spans: {len(span_indices)})")
            
            # STEP 3: Classify spans
            print("\n[6.4] Span Classification (FFNN):")
            spans_probability = model.span_ffnn(spans)
            print(f"  Span probability tensor shape: {spans_probability.shape}")
            print(f"  Each span is classified into one of {spans_probability.shape[-1]} categories:")
            for i, label in enumerate(SpanLabel):
                print(f"    {i}: {label.name}")
            
            # Show top predicted spans for each category
            print("\n  Top predicted spans:")
            for label_idx in range(1, spans_probability.shape[-1]):  # Skip INVALID class
                label_name = SpanLabel(label_idx).name
                top_indices = torch.topk(spans_probability[0, :, label_idx], min(3, spans_probability.shape[1])).indices.cpu().numpy()
                
                print(f"\n    Top {label_name} spans:")
                for rank, idx in enumerate(top_indices):
                    span = span_indices[idx]
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][span[0]:span[1]+1])
                    span_text = tokenizer.convert_tokens_to_string(tokens)
                    prob = spans_probability[0, idx, label_idx].item()
                    print(f"      {rank+1}: ({span[0]},{span[1]}) -> \"{span_text}\" (prob: {prob:.4f})")
            
            # STEP 4: Prune spans
            print("\n[6.5] Span Pruning:")
            print(f"  Pruning spans based on probability threshold {model.span_pruned_threshold}")
            print(f"  This step reduces computational complexity by selecting only high-probability spans")
            
            # Calculate the number of spans to keep based on pruning threshold
            batch_max_seq_len = max(seq_len)
            nz = int(batch_max_seq_len * model.span_pruned_threshold)
            target_indices, opinion_indices = model.pruned_target_opinion(spans_probability, nz)
            
            print(f"  Keeping top {nz} aspect and opinion spans")
            print(f"  Target indices tensor shape: {target_indices.shape}")
            print(f"  Opinion indices tensor shape: {opinion_indices.shape}")
            
            # Show pruned spans
            print("\n  Top pruned aspect spans:")
            for i in range(min(3, target_indices.shape[1])):
                idx = target_indices[0, i].item()
                span = span_indices[idx]
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0][span[0]:span[1]+1])
                span_text = tokenizer.convert_tokens_to_string(tokens)
                prob = spans_probability[0, idx, SpanLabel.ASPECT.value].item()
                print(f"    {i+1}: ({span[0]},{span[1]}) -> \"{span_text}\" (prob: {prob:.4f})")
            
            print("\n  Top pruned opinion spans:")
            for i in range(min(3, opinion_indices.shape[1])):
                idx = opinion_indices[0, i].item()
                span = span_indices[idx]
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0][span[0]:span[1]+1])
                span_text = tokenizer.convert_tokens_to_string(tokens)
                prob = spans_probability[0, idx, SpanLabel.OPINION.value].item()
                print(f"    {i+1}: ({span[0]},{span[1]}) -> \"{span_text}\" (prob: {prob:.4f})")
            
            # STEP 5: Form target-opinion pairs
            print("\n[6.6] Target-Opinion Pair Representation:")
            candidates, candidate_indices, relation_indices = model.target_opinion_pair_representation(
                spans, span_indices, target_indices, opinion_indices)
            
            print(f"  Candidate representation tensor shape: {candidates.shape}")
            print(f"  Each candidate pair is represented as [St; So; f_distance(St,So)] where:")
            print(f"    - St is the representation of the target span")
            print(f"    - So is the representation of the opinion span")
            print(f"    - f_distance(St,So) is a learned embedding of the distance between spans")
            
            # Show some candidate pairs
            print("\n  Example candidate pairs:")
            for i in range(min(5, len(candidate_indices[0]))):
                a, b, c, d = candidate_indices[0][i]
                aspect_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][a:b+1])
                aspect_text = tokenizer.convert_tokens_to_string(aspect_tokens)
                
                opinion_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][c:d+1])
                opinion_text = tokenizer.convert_tokens_to_string(opinion_tokens)
                
                distance = model.target_opinion_pair_representation.min_distance(a, b, c, d)
                print(f"    Pair {i+1}: ({a},{b},{c},{d})")
                print(f"      Aspect: \"{aspect_text}\"")
                print(f"      Opinion: \"{opinion_text}\"")
                print(f"      Min distance: {distance} tokens")
            print(f"    ... (total candidate pairs: {len(candidate_indices[0])})")
            
            # STEP 6: Classify relations
            print("\n[6.7] Relation Classification (FFNN):")
            relations_probability = model.pairs_ffnn(candidates)
            print(f"  Relation probability tensor shape: {relations_probability.shape}")
            print(f"  Each relation is classified into one of {relations_probability.shape[-1]} categories:")
            for i, label in enumerate(RelationLabel):
                print(f"    {i}: {label.name}")
            
            # Show top predicted triplets
            print("\n  Top predicted sentiment triplets:")
            # Get indices of relations with highest non-INVALID probability
            non_invalid_prob = relations_probability[0, :, 1:].sum(dim=1)
            top_relation_indices = torch.topk(non_invalid_prob, min(5, relations_probability.shape[1])).indices.cpu().numpy()
            
            for rank, idx in enumerate(top_relation_indices):
                if idx < len(candidate_indices[0]):
                    a, b, c, d = candidate_indices[0][idx]
                    aspect_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][a:b+1])
                    aspect_text = tokenizer.convert_tokens_to_string(aspect_tokens)
                    
                    opinion_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][c:d+1])
                    opinion_text = tokenizer.convert_tokens_to_string(opinion_tokens)
                    
                    # Get sentiment with highest probability (excluding INVALID)
                    sentiment_probs = relations_probability[0, idx].cpu().numpy()
                    sentiment_idx = np.argmax(sentiment_probs[1:]) + 1  # +1 because we skipped INVALID
                    sentiment = RelationLabel(sentiment_idx).name
                    prob = sentiment_probs[sentiment_idx]
                    
                    print(f"    Triplet {rank+1}:")
                    print(f"      Aspect: \"{aspect_text}\"")
                    print(f"      Opinion: \"{opinion_text}\"")
                    print(f"      Sentiment: {sentiment} (prob: {prob:.4f})")
            
            print("\n" + "="*80)
            print(f"PROCESSING COMPLETED FOR EXAMPLE {batch_idx+1}")
            print("="*80)
            
            # Process only the first batch for clarity
            if batch_idx == 0:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SPAN-ASTE model processing")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--data_path", required=True, help="Path to data directory containing dev_triplets.txt")
    parser.add_argument("--bert_model", default="bert-base-uncased", help="BERT model name or path")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(1024)
    
    # Run debug visualization
    debug_visualize_aste_model(
        args.model_path,
        args.data_path,
        args.bert_model,
        args.max_seq_len
    )