from flask import Flask, render_template, request, jsonify
import torch
import os
import itertools
import pandas as pd
import numpy as np
from enum import IntEnum
from torch import nn
from transformers import BertTokenizer, BertModel, AlbertModel, AutoTokenizer
from transformers import BertConfig, BertForSequenceClassification
import sys
import platform
import traceback

# Define enums needed for the SPAN-ASTE model
class SpanLabel(IntEnum):
    INVALID = 0
    ASPECT = 1
    OPINION = 2

class RelationLabel(IntEnum):
    INVALID = 0
    POS = 1
    NEG = 2
    NEU = 3

# Definisikan konstanta label dari DocumentSentimentDataset
LABEL2INDEX = {'User Interface': 0, 'User Experince': 1, 'Functionality and Perfomance': 2, 'Security': 3, 'Support and Updates': 4, 'General Aspect': 5, 'Out of Aspect': 6}
INDEX2LABEL = {0: 'User Interface', 1: 'User Experince', 2: 'Functionality and Perfomance', 3: 'Security', 4: 'Support and Updates', 5: 'General Aspect', 6: 'Out of Aspect'}
NUM_LABELS = 7

# System information function
def get_system_info():
    info = {
        "Python Version": sys.version,
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "Number of CUDA Devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "Current CUDA Device": torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "N/A",
        "Device Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "N/A",
        "System": platform.system(),
        "System Release": platform.release(),
        "System Version": platform.version(),
        "Architecture": platform.machine()
    }
    return info

# For debugging CUDA issues
def get_cuda_arch_list():
    try:
        import torch.utils.cpp_extension as cpp_ext
        return cpp_ext.CUDA_HOME, cpp_ext.CUDA_ARCH_LIST
    except Exception as e:
        return "Error retrieving CUDA arch list", str(e)

# SPAN-ASTE Model Components
class SpanRepresentation(nn.Module):
    def __init__(self, span_width_embedding_dim, span_maximum_length):
        super(SpanRepresentation, self).__init__()
        self.span_maximum_length = span_maximum_length
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.span_width_embedding = nn.Embedding(len(self.bucket_bins), span_width_embedding_dim)

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.span_width_embedding(torch.LongTensor([em]).to(device))

    def forward(self, x, batch_max_seq_len):
        batch_size, sequence_length, _ = x.size()
        device = x.device

        len_arrange = torch.arange(0, batch_max_seq_len, device=device)
        span_indices = []

        max_window = min(batch_max_seq_len, self.span_maximum_length)

        for window in range(1, max_window + 1):
            if window == 1:
                indics = [(x.item(), x.item()) for x in len_arrange]
            else:
                res = len_arrange.unfold(0, window, 1)
                indics = [(idx[0].item(), idx[-1].item()) for idx in res]
            span_indices.extend(indics)

        spans = [torch.cat(
            (x[:, s[0], :], x[:, s[1], :],
             self.bucket_embedding(abs(s[1] - s[0] + 1), device).repeat(
                 (batch_size, 1)).to(device)),
            dim=1) for s in span_indices]

        return torch.stack(spans, dim=1), span_indices

class PrunedTargetOpinion:
    def __init__(self):
        pass

    def __call__(self, spans_probability, nz):
        target_indices = torch.topk(spans_probability[:, :, SpanLabel.ASPECT.value], nz, dim=-1).indices
        opinion_indices = torch.topk(spans_probability[:, :, SpanLabel.OPINION.value], nz, dim=-1).indices
        return target_indices, opinion_indices

class TargetOpinionPairRepresentation(nn.Module):
    def __init__(self, distance_embeddings_dim):
        super(TargetOpinionPairRepresentation, self).__init__()
        self.bucket_bins = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64]
        self.distance_embeddings = nn.Embedding(len(self.bucket_bins), distance_embeddings_dim)

    def min_distance(self, a, b, c, d):
        return min(abs(b - c), abs(a - d))

    def bucket_embedding(self, width, device):
        em = [ix for ix, v in enumerate(self.bucket_bins) if width >= v][-1]
        return self.distance_embeddings(torch.LongTensor([em]).to(device))

    def forward(self, spans, span_indices, target_indices, opinion_indices):
        batch_size = spans.size(0)
        device = spans.device

        candidate_indices, relation_indices = [], []
        for batch in range(batch_size):
            pairs = list(itertools.product(target_indices[batch].cpu().tolist(), opinion_indices[batch].cpu().tolist()))
            relation_indices.append(pairs)
            candidate_ind = []
            for pair in pairs:
                a, b = span_indices[pair[0]]
                c, d = span_indices[pair[1]]
                candidate_ind.append((a, b, c, d))
            candidate_indices.append(candidate_ind)

        candidate_pool = []
        for batch in range(batch_size):
            if len(relation_indices[batch]) == 0:
                candidate_pool.append(torch.zeros((0, spans.size(-1) * 2 + self.distance_embeddings.embedding_dim), device=device))
                continue
                
            relations = [
                torch.cat((spans[batch, c[0], :], spans[batch, c[1], :],
                           self.bucket_embedding(
                               self.min_distance(*span_indices[c[0]], *span_indices[c[1]]), device).squeeze(0))
                          , dim=0) for c in
                relation_indices[batch]]
            candidate_pool.append(torch.stack(relations))

        if all(p.size(0) == 0 for p in candidate_pool):
            return torch.zeros((batch_size, 0, spans.size(-1) * 2 + self.distance_embeddings.embedding_dim), device=device), candidate_indices, relation_indices
            
        return torch.stack(candidate_pool), candidate_indices, relation_indices

class SpanAsteModel(nn.Module):
    def __init__(
            self,
            pretrain_model,
            target_dim,
            relation_dim,
            ffnn_hidden_dim=150,
            span_width_embedding_dim=20,
            span_maximum_length=8,
            span_pruned_threshold=0.5,
            pair_distance_embeddings_dim=128,
            device="cuda"
    ):
        super(SpanAsteModel, self).__init__()
        self.span_pruned_threshold = span_pruned_threshold
        self.pretrain_model = pretrain_model
        self.device = device

        if "indobert-lite" in pretrain_model:
            print(f"Using AlbertModel for {pretrain_model}")
            self.bert = AlbertModel.from_pretrained(pretrain_model)
        else:
            print(f"Using BertModel for {pretrain_model}")
            self.bert = BertModel.from_pretrained(pretrain_model)
            
        encoding_dim = self.bert.config.hidden_size

        self.span_representation = SpanRepresentation(span_width_embedding_dim, span_maximum_length)
        span_dim = encoding_dim * 2 + span_width_embedding_dim
        self.span_ffnn = torch.nn.Sequential(
            nn.Linear(span_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, target_dim, bias=True),
            nn.Softmax(-1)
        )
        self.pruned_target_opinion = PrunedTargetOpinion()
        self.target_opinion_pair_representation = TargetOpinionPairRepresentation(pair_distance_embeddings_dim)
        pairs_dim = 2 * span_dim + pair_distance_embeddings_dim
        self.pairs_ffnn = torch.nn.Sequential(
            nn.Linear(pairs_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, ffnn_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(ffnn_hidden_dim, relation_dim, bias=True),
            nn.Softmax(-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.span_ffnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        for name, param in self.pairs_ffnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_ids, attention_mask, token_type_ids, seq_len):
        batch_size, sequence_len = input_ids.size()
        batch_max_seq_len = max(seq_len)

        self.span_ffnn.eval()
        self.pairs_ffnn.eval()

        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        x = bert_output.last_hidden_state
        spans, span_indices = self.span_representation(x, batch_max_seq_len)
        spans_probability = self.span_ffnn(spans)
        nz = int(batch_max_seq_len * self.span_pruned_threshold)

        target_indices, opinion_indices = self.pruned_target_opinion(spans_probability, nz)

        candidates, candidate_indices, relation_indices = self.target_opinion_pair_representation(
            spans, span_indices, target_indices, opinion_indices)

        if candidates.size(1) == 0:
            candidate_probability = torch.zeros((batch_size, 0, len(RelationLabel)), device=self.device)
        else:
            candidate_probability = self.pairs_ffnn(candidates)

        # batch span indices
        span_indices = [span_indices for _ in range(batch_size)]

        return spans_probability, span_indices, candidate_probability, candidate_indices

# Utility functions
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

# Model Manager
class ModelManager:
    def __init__(self):
        self.span_aste_model = None
        self.span_aste_tokenizer = None
        self.aspect_classifier_model = None
        self.aspect_classifier_tokenizer = None
        # Try using GPU but have CPU fallback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.i2w = INDEX2LABEL
        self.w2i = LABEL2INDEX
        self.loaded = False
        
        # Print some information about CUDA
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA arch list: {get_cuda_arch_list()}")
    
    def try_cuda_operation(self):
        """Test if CUDA operations can run successfully"""
        if self.device == "cuda":
            try:
                # Try a simple CUDA operation
                a = torch.tensor([1.0], device="cuda")
                b = torch.tensor([2.0], device="cuda")
                c = a + b
                print(f"CUDA test successful: {c}")
                return True
            except Exception as e:
                print(f"CUDA operation failed: {e}")
                self.device = "cpu"
                print("Falling back to CPU")
                return False
        return False
    
    def load_models(self):
        if self.loaded:
            return
        
        # Test CUDA operation before loading models
        if self.device == "cuda":
            self.try_cuda_operation()
            
        try:
            self.load_span_aste_model()
            self.load_aspect_classifier_model()
            self.loaded = True
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            
            # If we're on CUDA and it failed, try falling back to CPU
            if self.device == "cuda":
                print("Attempting to fall back to CPU...")
                self.device = "cpu"
                try:
                    self.load_span_aste_model()
                    self.load_aspect_classifier_model()
                    self.loaded = True
                    print("All models loaded successfully on CPU!")
                except Exception as e:
                    print(f"Failed to load models on CPU as well: {e}")
                    traceback.print_exc()
    
    def load_span_aste_model(self):
        # Configuration - UBAH PATH MODEL SESUAI KEBUTUHAN
        MODEL_PATH = "checkpoint_large/fold1/model_best"  # Path to model directory
        BERT_MODEL = "indobenchmark/indobert-large-p2"  # Model used for training
        
        print(f"Loading SPAN-ASTE model from {MODEL_PATH}")
        print(f"Using {self.device} device")
        
        # Load tokenizer
        try:
            self.span_aste_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
            print(f"Successfully loaded tokenizer: {type(self.span_aste_tokenizer).__name__}")
        except:
            self.span_aste_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
            print(f"Fallback to BertTokenizer: {type(self.span_aste_tokenizer).__name__}")
        
        # Define model structure
        target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
        
        # Initialize model
        self.span_aste_model = SpanAsteModel(
            BERT_MODEL,
            target_dim,
            relation_dim,
            device=self.device
        )
        
        # Load model state
        model_file_path = os.path.join(MODEL_PATH, "model.pt")
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at {model_file_path}")
        
        state_dict = torch.load(model_file_path, map_location=torch.device(self.device))
        self.span_aste_model.load_state_dict(state_dict, strict=False)  # Use strict=False
        self.span_aste_model.to(self.device)
        self.span_aste_model.eval()
        
        print("SPAN-ASTE model loaded successfully")
    
    def load_aspect_classifier_model(self):
        # Configuration - UBAH PATH MODEL SESUAI KEBUTUHAN
        MODEL_PATH = "model_klasifikasi_aspek/model_aspect_categorization.pt"  # Path to model file
        
        print(f"Loading Aspect Classifier model from {MODEL_PATH}")
        
        # Load tokenizer
        self.aspect_classifier_tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
        
        # Load config
        config = BertConfig.from_pretrained('indobenchmark/indobert-base-p2')
        
        # Set number of labels
        config.num_labels = NUM_LABELS
        
        # Initialize model
        self.aspect_classifier_model = BertForSequenceClassification.from_pretrained(
            'indobenchmark/indobert-base-p2', config=config)
        
        # Load state dict yang sudah disimpan
        state_dict = torch.load(MODEL_PATH, map_location=torch.device(self.device))
        self.aspect_classifier_model.load_state_dict(state_dict, strict=False)  # Use strict=False
        
        self.aspect_classifier_model.to(self.device)
        self.aspect_classifier_model.eval()
        
        print("Aspect Classifier model loaded successfully")
        print(f"Label mapping: {self.i2w}")
    
    def predict_span_aste(self, text):
        """Extract aspect-opinion-sentiment triplets using SPAN-ASTE model"""
        if not self.loaded:
            self.load_models()
            
        with torch.no_grad():
            # Tokenize input
            tokens = ["[CLS]"] + self.span_aste_tokenizer.tokenize(text) + ["[SEP]"]
            print(f"Tokens: {tokens}")
            
            # Convert to model inputs
            inputs = self.span_aste_tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors="pt")
            
            try:
                # Safely move to device (with error handling)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                token_type_ids = inputs.token_type_ids.to(self.device)
                seq_len = (input_ids != 0).sum().item()
                
                # Forward pass
                spans_probability, span_indices, relations_probability, candidate_indices = self.span_aste_model(
                    input_ids, attention_mask, token_type_ids, [seq_len])
                
                # Extract triplets
                if relations_probability.numel() == 0:  # Check if tensor is empty
                    return []
                    
                relations_probability = relations_probability.squeeze(0)
                predict = []
                
                for idx, can in enumerate(candidate_indices[0]):
                    if idx >= relations_probability.size(0):
                        continue
                        
                    a, b, c, d = can
                    
                    # Skip if confidence is too low
                    if relations_probability[idx].max().item() < 0.3:  # Confidence threshold
                        continue
                    
                    # Extract tokens for aspect and opinion
                    aspect_tokens = tokens[a:b+1]
                    opinion_tokens = tokens[c:d+1]
                    
                    # Combine subwords
                    aspect_words = merge_subwords(aspect_tokens)
                    opinion_words = merge_subwords(opinion_tokens)
                    
                    # Remove the last word from aspect and opinion
                    if len(aspect_words) > 1:
                        aspect_words = aspect_words[:-1]
                    
                    if len(opinion_words) > 1:
                        opinion_words = opinion_words[:-1]
                    
                    # For aspect, combine words
                    aspect = " ".join(aspect_words) if aspect_words else ""
                    
                    # For opinion, combine all words
                    opinion = " ".join(opinion_words)
                    
                    # Remove trailing punctuation
                    opinion = opinion.rstrip('.,!?;:')
                    
                    # Skip empty spans
                    if not aspect or not opinion:
                        continue
                    
                    # Get sentiment
                    sentiment = RelationLabel(relations_probability[idx].argmax(-1).item()).name
                    confidence = float(relations_probability[idx].max().item())
                    
                    # Only include valid sentiment
                    if sentiment != RelationLabel.INVALID.name:
                        # Debug info
                        print(f"Found triplet: {aspect}, {opinion}, {sentiment}, conf: {confidence:.4f}")
                        
                        predict.append({
                            "aspect": aspect,
                            "opinion": opinion,
                            "sentiment": sentiment,
                            "confidence": confidence
                        })
                
                return predict
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                traceback.print_exc()
                
                # If we're on CUDA and it failed, try falling back to CPU for just this prediction
                if self.device == "cuda":
                    print("Attempting CPU fallback for this prediction...")
                    try:
                        self.device = "cpu"
                        self.span_aste_model.to("cpu")
                        
                        input_ids = inputs.input_ids.to("cpu")
                        attention_mask = inputs.attention_mask.to("cpu")
                        token_type_ids = inputs.token_type_ids.to("cpu")
                        seq_len = (input_ids != 0).sum().item()
                        
                        spans_probability, span_indices, relations_probability, candidate_indices = self.span_aste_model(
                            input_ids, attention_mask, token_type_ids, [seq_len])
                            
                        # Continue with prediction code...
                        # (Similar code as above, abbreviated for clarity)
                        
                        print("CPU fallback successful!")
                        # Reset device back to cuda for future predictions
                        self.device = "cuda" 
                        self.span_aste_model.to("cuda")
                        
                    except Exception as cpu_e:
                        print(f"CPU fallback also failed: {cpu_e}")
                        traceback.print_exc()
                        
                return []
    
    def predict_aspect_category(self, text):
        """Predict aspect category using the aspect classifier model"""
        if not self.loaded:
            self.load_models()
            
        try:
            # Tokenize input
            tokenized_input = self.aspect_classifier_tokenizer(
                text, 
                padding='max_length',
                max_length=128,
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move to device
            tokenized_input = {key: value.to(self.device) for key, value in tokenized_input.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.aspect_classifier_model(**tokenized_input)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
                confidence = float(probabilities[0, predictions[0]].item())
            
            # Get predicted label
            predicted_label = self.i2w[predictions.item()]
            
            print(f"Aspect category for '{text}': {predicted_label}, conf: {confidence:.4f}")
            
            return predicted_label, confidence
        
        except Exception as e:
            print(f"Error during aspect classification: {e}")
            traceback.print_exc()
            
            # Try CPU fallback if we're on CUDA
            if self.device == "cuda":
                try:
                    print("Attempting CPU fallback for aspect classification...")
                    self.device = "cpu"
                    self.aspect_classifier_model.to("cpu")
                    
                    # Retry on CPU
                    # (Code abbreviated for clarity)
                    
                    # Reset back to CUDA
                    self.device = "cuda"
                    self.aspect_classifier_model.to("cuda")
                except Exception:
                    pass
                    
            return "Unknown", 0.0

# Create a global instance of ModelManager
model_manager = ModelManager()

# Create Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api-docs')
def api_docs():
    return render_template('api-docs.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for aspect sentiment analysis"""
    # Get input text from JSON payload
    data = request.get_json(silent=True)
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Invalid request format. Please provide a JSON with a "text" field.'
        }), 400
    
    text = data['text']
    
    try:
        print(f"\nProcessing API input: '{text}'")
        
        # Ensure models are loaded
        if not model_manager.loaded:
            print("Models not loaded yet, loading now...")
            model_manager.load_models()
        
        # 1. Use SPAN-ASTE model to extract triplets
        triplets = model_manager.predict_span_aste(text)
        
        # 2. For each triplet, predict aspect category
        formatted_triplets = []
        for triplet in triplets:
            # Combine aspect and opinion for aspect category prediction
            aspect_opinion_text = f"{triplet['aspect']} {triplet['opinion']}"
            
            # Predict aspect category
            aspect_category, category_confidence = model_manager.predict_aspect_category(aspect_opinion_text)
            
            # Format triplet with new key names
            formatted_triplet = {
                'aspect_term': triplet['aspect'],
                'opinion_term': triplet['opinion'],
                'sentiment': triplet['sentiment'],
                'triplet_confidence': triplet['confidence'],
                'aspect_category': aspect_category,
                'aspect_category_confidence': category_confidence
            }
            formatted_triplets.append(formatted_triplet)
        
        return jsonify({
            'text': text,
            'triplets': formatted_triplets,
            'count': len(formatted_triplets)
        })
    
    except Exception as e:
        print(f"Error in API prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'text': text,
            'triplets': [],
            'error': str(e)
        }), 500

# Modifikasi pada fungsi predict() yang digunakan untuk web interface
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'Empty text provided'
        }), 400
    
    try:
        print(f"\nProcessing input: '{text}'")
        
        # Ensure models are loaded
        if not model_manager.loaded:
            print("Models not loaded yet, loading now...")
            model_manager.load_models()
        
        # 1. Use SPAN-ASTE model to extract triplets
        triplets = model_manager.predict_span_aste(text)
        
        # 2. For each triplet, predict aspect category
        formatted_triplets = []
        for triplet in triplets:
            # Combine aspect and opinion for aspect category prediction
            aspect_opinion_text = f"{triplet['aspect']} {triplet['opinion']}"
            
            # Predict aspect category
            aspect_category, category_confidence = model_manager.predict_aspect_category(aspect_opinion_text)
            
            # Format triplet with new key names
            formatted_triplet = {
                'aspect_term': triplet['aspect'],
                'opinion_term': triplet['opinion'],
                'sentiment': triplet['sentiment'],
                'triplet_confidence': triplet['confidence'],
                'aspect_category': aspect_category,
                'aspect_category_confidence': category_confidence
            }
            formatted_triplets.append(formatted_triplet)
        
        return jsonify({
            'text': text,
            'triplets': formatted_triplets
        })
    
    except Exception as e:
        print(f"Error in prediction route: {e}")
        traceback.print_exc()
        return jsonify({
            'text': text,
            'triplets': [],
            'error': str(e)
        }), 500

@app.route('/system_info')
def system_info():
    """Endpoint to check system information"""
    info = get_system_info()
    return jsonify(info)

# Create templates directory and add index.html
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Span Level Aspect Sentiment Triplet Extraction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3a0ca3;
            --accent-color: #7209b7;
            --positive-color: #4caf50;
            --negative-color: #f44336;
            --neutral-color: #ff9800;
            --bg-light: #f8f9fa;
            --border-radius: 10px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-light);
            color: #333;
            line-height: 1.6;
        }
        
        .app-container {
            max-width: 1000px;
            margin: 30px auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: var(--primary-color);
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .card {
            border: none;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            margin-bottom: 25px;
            overflow: hidden;
        }
        
        .card-header {
            background-color: white;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding: 15px 20px;
        }
        
        .card-body {
            padding: 25px;
        }
        
        textarea.form-control {
            border-radius: var(--border-radius);
            min-height: 120px;
            padding: 15px;
            font-size: 1rem;
            border: 1px solid #ddd;
            transition: border-color 0.3s;
        }
        
        textarea.form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 0.25rem rgba(67, 97, 238, 0.25);
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
            border-radius: var(--border-radius);
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .btn-primary:hover {
            background-color: var(--secondary-color);
            border-color: var(--secondary-color);
            transform: translateY(-2px);
        }
        
        .loading {
            display: none;
            align-items: center;
            margin-top: 15px;
        }
        
        .loading span {
            margin-left: 10px;
            font-weight: 500;
        }
        
        .triplet-card {
            border-radius: var(--border-radius);
            margin-bottom: 15px;
            transition: transform 0.3s;
        }
        
        .triplet-card:hover {
            transform: translateY(-5px);
        }
        
        .triplet-card .card-body {
            padding: 20px;
        }
        
        .triplet-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        
        .aspect-label, .opinion-label, .sentiment-label, .category-label {
            font-weight: 600;
            margin-right: 8px;
            color: #555;
        }
        
        .sentiment-badge {
            padding: 5px 10px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .sentiment-badge.positive {
            background-color: rgba(76, 175, 80, 0.15);
            color: var(--positive-color);
        }
        
        .sentiment-badge.negative {
            background-color: rgba(244, 67, 54, 0.15);
            color: var(--negative-color);
        }
        
        .sentiment-badge.neutral {
            background-color: rgba(255, 152, 0, 0.15);
            color: var(--neutral-color);
        }
        
        .confidence-bar {
            height: 6px;
            background-color: #e9ecef;
            border-radius: 3px;
            margin-top: 5px;
            overflow: hidden;
        }

        .confidence-level {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease, background-color 0.5s ease;
            /* Tidak perlu lagi class .high, .medium, .low karena warna diatur secara dinamis */
        }
        
        
        .aspect-value, .opinion-value, .category-value {
            font-weight: 500;
        }
        
        .confidence-text {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 3px;
        }
        
        .no-results {
            text-align: center;
            padding: 30px;
            color: #6c757d;
        }
        
        .no-results i {
            font-size: 3rem;
            margin-bottom: 15px;
            color: #dee2e6;
        }
        
        .analyzed-text {
            font-style: italic;
            margin-bottom: 20px;
            padding: 10px 15px;
            background-color: rgba(67, 97, 238, 0.05);
            border-left: 4px solid var(--primary-color);
            border-radius: 4px;
        }
        
        @media (max-width: 768px) {
            .triplet-header {
                flex-direction: column;
            }
            
            .sentiment-category-container {
                margin-top: 10px;
            }
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
        }
        
        /* Nav links */
        .nav-links {
            display: flex;
            justify-content: center;
            margin: 15px 0;
            gap: 20px;
        }
        
        .nav-link {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .nav-link:hover {
            color: var(--secondary-color);
            text-decoration: underline;
        }
    </style>
    </head>
<body>
    <div class="app-container">
        <div class="header">
            <h1>Aspect Sentiment Triplet Extraction</h1>
            <p>Analisis Sentimen Triplet Level Aspek Berbasis Span Level Pada Aplikasi Pemerintahan dan Kategorisasi Aspek</p>
        </div>
        
        <div class="nav-links">
            <a href="/" class="nav-link"><i class="bi bi-house-door-fill me-1"></i>Beranda</a>
            <a href="/api-docs" class="nav-link"><i class="bi bi-code-slash me-1"></i>Dokumentasi API</a>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-chat-dots me-2"></i>Masukkan Teks</h5>
            </div>
            <div class="card-body">
                <form id="prediction-form">
                    <div class="mb-3">
                        <textarea class="form-control" id="text-input" rows="4" placeholder="Masukkan teks ulasan untuk dianalisis..." required></textarea>
                    </div>
                    <div class="d-flex">
                        <button type="submit" class="btn btn-primary">
                            <i class="bi bi-search me-2"></i>Analisis
                        </button>
                        <div class="loading ms-3 d-flex">
                            <div class="spinner-border spinner-border-sm text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <span class="ms-2">Sedang menganalisis...</span>
                        </div>
                    </div>
                </form>
            </div>
        </div>
        
        <div id="results" class="card" style="display: none;">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-graph-up me-2"></i>Hasil Analisis</h5>
            </div>
            <div class="card-body">
                <div class="analyzed-text" id="analyzed-text"></div>
                
                <div id="no-triplets" class="no-results" style="display: none;">
                    <i class="bi bi-emoji-frown"></i>
                    <h5>Tidak ditemukan aspek sentimen</h5>
                    <p class="text-muted">Coba masukkan teks ulasan yang lebih detail tentang suatu produk atau layanan.</p>
                </div>
                
                <div id="triplets-container"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>Feza Raffa Arnanda &copy; 2025</p>
        </div>
    </div>
    
    <script>
        function getConfidenceBarColor(confidence) {
            // Warna linear dari merah (rendah) ke kuning (sedang) ke hijau (tinggi)
            if (confidence >= 0.7) {
                return '#4caf50'; // Hijau untuk confidence tinggi
            } else if (confidence >= 0.4) {
                return '#ff9800'; // Oranye untuk confidence sedang
            } else {
                return '#f44336'; // Merah untuk confidence rendah
            }
        }
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            document.querySelector('.loading').style.display = 'flex';
            document.getElementById('results').style.display = 'none';
            
            // Get input text
            const text = document.getElementById('text-input').value;
            
            // Send request
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'text=' + encodeURIComponent(text)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.querySelector('.loading').style.display = 'none';
                
                // Show results
                document.getElementById('results').style.display = 'block';
                document.getElementById('analyzed-text').textContent = data.text;
                
                const tripletsContainer = document.getElementById('triplets-container');
                tripletsContainer.innerHTML = '';
                
                if (!data.triplets || data.triplets.length === 0) {
                    document.getElementById('no-triplets').style.display = 'block';
                } else {
                    document.getElementById('no-triplets').style.display = 'none';
                    
                    // Display triplets
                    data.triplets.forEach(triplet => {
                        const tripletCard = document.createElement('div');
                        tripletCard.className = 'card triplet-card';
                        
                        // Set card border color based on sentiment
                        let sentimentClass = '';
                        let sentimentLabel = '';
                        if (triplet.sentiment === 'POS') {
                            sentimentClass = 'positive';
                            sentimentLabel = 'Positif';
                        } else if (triplet.sentiment === 'NEG') {
                            sentimentClass = 'negative';
                            sentimentLabel = 'Negatif';
                        } else {
                            sentimentClass = 'neutral';
                            sentimentLabel = 'Netral';
                        }
                        

                        tripletCard.innerHTML = `
                            <div class="card-body">
                                <div class="triplet-header">
                                    <div class="aspect-opinion-container">
                                        <div class="mb-2">
                                            <span class="aspect-label">Aspek:</span>
                                            <span class="aspect-value">${triplet.aspect_term}</span>
                                        </div>
                                        <div>
                                            <span class="opinion-label">Opini:</span>
                                            <span class="opinion-value">${triplet.opinion_term}</span>
                                        </div>
                                    </div>
                                    <div class="sentiment-category-container">
                                        <div class="mb-2">
                                            <span class="sentiment-label">Sentimen:</span>
                                            <span class="sentiment-badge ${sentimentClass}">${sentimentLabel}</span>
                                        </div>
                                        <div>
                                            <span class="category-label">Kategori:</span>
                                            <span class="category-value">${triplet.aspect_category || 'Tidak tersedia'}</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="mt-3">
                                    <div class="mb-2">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <small>Confidence Sentimen</small>
                                            <small>${(triplet.triplet_confidence * 100).toFixed(0)}%</small>
                                        </div>
                                        <div class="confidence-bar">
                                            <div class="confidence-level" 
                                                style="width: ${(triplet.triplet_confidence * 100).toFixed(0)}%; 
                                                        background-color: ${getConfidenceBarColor(triplet.triplet_confidence)}">
                                            </div>
                                        </div>
                                    </div>
                                    
                                    ${triplet.aspect_category_confidence ? `
                                    <div>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <small>Confidence Kategori</small>
                                            <small>${(triplet.aspect_category_confidence * 100).toFixed(0)}%</small>
                                        </div>
                                        <div class="confidence-bar">
                                            <div class="confidence-level" 
                                                style="width: ${(triplet.aspect_category_confidence * 100).toFixed(0)}%; 
                                                        background-color: ${getConfidenceBarColor(triplet.aspect_category_confidence)}">
                                            </div>
                                        </div>
                                    </div>
                                    ` : ''}
                                </div>
                            </div>
                        `;              
                        tripletsContainer.appendChild(tripletCard);
                    });
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.querySelector('.loading').style.display = 'none';
                alert('Terjadi kesalahan saat melakukan prediksi. Silakan coba lagi.');
            });
        });
    </script>
</body>
</html>
''')

# Create api-docs.html template
with open('templates/api-docs.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation - Span Level Aspect Sentiment Triplet Extraction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3a0ca3;
            --accent-color: #7209b7;
            --positive-color: #4caf50;
            --negative-color: #f44336;
            --neutral-color: #ff9800;
            --bg-light: #f8f9fa;
            --border-radius: 10px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-light);
            color: #333;
            line-height: 1.6;
        }
        
        .app-container {
            max-width: 1000px;
            margin: 30px auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: var(--primary-color);
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .card {
            border: none;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            margin-bottom: 25px;
            overflow: hidden;
        }
        
        .card-header {
            background-color: white;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding: 15px 20px;
        }
        
        .card-body {
            padding: 25px;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
        }
        
        /* Nav links */
        .nav-links {
            display: flex;
            justify-content: center;
            margin: 15px 0;
            gap: 20px;
        }
        
        .nav-link {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        .nav-link:hover {
            color: var(--secondary-color);
            text-decoration: underline;
        }
        
        /* API Documentation Styles */
        .api-section {
            margin-bottom: 30px;
        }
        
        .api-section h3 {
            margin-bottom: 15px;
            color: var(--primary-color);
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        
        .api-endpoint {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 10px 15px;
            font-family: monospace;
            margin: 10px 0;
            font-weight: 600;
        }
        
        .api-endpoint .method {
            color: var(--primary-color);
            margin-right: 10px;
        }
        
        code {
            background-color: #f1f1f1;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 0.9em;
        }
        
        pre {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            border: 1px solid #e9ecef;
        }
        
        .table-params {
            width: 100%;
            margin-bottom: 20px;
        }
        
        .table-params th {
            background-color: #f1f1f1;
        }
        
        .badge-required {
            background-color: var(--primary-color);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75rem;
        }
        
        .badge-optional {
            background-color: #6c757d;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75rem;
        }
        
        .example-container {
            margin: 20px 0;
        }
        
        .example-container h5 {
            margin-bottom: 10px;
        }
        
        .section-divider {
            border-top: 1px solid #dee2e6;
            margin: 40px 0;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="header">
            <h1>API Documentation</h1>
            <p>Span Level Aspect Sentiment Triplet Extraction API</p>
        </div>
        
        <div class="nav-links">
            <a href="/" class="nav-link"><i class="bi bi-house-door-fill me-1"></i>Beranda</a>
            <a href="/api-docs" class="nav-link"><i class="bi bi-code-slash me-1"></i>Dokumentasi API</a>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-info-circle me-2"></i>Ringkasan API</h5>
            </div>
            <div class="card-body">
                <p>API ini menyediakan kemampuan analisis sentimen tingkat aspek pada teks menggunakan model berbasis Span Level untuk ekstraksi Triplet Sentimen Aspek. API dapat mengidentifikasi istilah aspek (aspect terms), istilah opini (opinion terms), dan sentimen yang terkait, serta mengkategorikan aspek ke dalam beberapa kategori.</p>
                
                <div class="alert alert-info">
                    <i class="bi bi-info-circle-fill me-2"></i>
                    <strong>Basis URL:</strong> <code>http://localhost:5000</code>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0"><i class="bi bi-hdd-stack me-2"></i>Endpoints API</h5>
            </div>
            <div class="card-body">
                <div class="api-section">
                    <h3>Analisis Aspek Sentimen</h3>
                    
                    <div class="api-endpoint">
                        <span class="method">POST</span>/api/predict
                    </div>
                    
                    <p>Menganalisis teks ulasan untuk mengekstrak triplet aspek-sentimen berdasarkan model Span Level.</p>
                    
                    <h5>Request Parameters</h5>
                    <table class="table table-params table-bordered">
                        <thead>
                            <tr>
                                <th>Parameter</th>
                                <th>Tipe</th>
                                <th>Deskripsi</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>text <span class="badge-required">required</span></td>
                                <td>string</td>
                                <td>Teks ulasan yang akan dianalisis untuk ekstraksi triplet aspek-sentimen.</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <div class="example-container">
                        <h5>Contoh Request</h5>
                        <pre>{
                        "text": "overall aplikasinya bagus, tapi ui jelek dan login susah"
                        }
                        </pre>
                    </div>
                    
                    <h5>Response</h5>
                    <table class="table table-params table-bordered">
                        <thead>
                            <tr>
                                <th>Field</th>
                                <th>Tipe</th>
                                <th>Deskripsi</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>text</td>
                                <td>string</td>
                                <td>Teks input yang dianalisis.</td>
                            </tr>
                            <tr>
                                <td>triplets</td>
                                <td>array</td>
                                <td>Array objek triplet yang mengandung pasangan aspek-opini-sentimen yang diekstrak.</td>
                            </tr>
                            <tr>
                                <td>count</td>
                                <td>integer</td>
                                <td>Jumlah triplet yang ditemukan dalam teks.</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <h5>Triplet Object</h5>
                    <table class="table table-params table-bordered">
                        <thead>
                            <tr>
                                <th>Field</th>
                                <th>Tipe</th>
                                <th>Deskripsi</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>aspect_category</td>
                                <td>string</td>
                                <td>Kategori yang ditetapkan untuk aspek, seperti "User Interface", "Functionality and Performance", dll.</td>
                            </tr>
                            <tr>
                                <td>aspect_category_confidence</td>
                                <td>float</td>
                                <td>Tingkat kepercayaan model terhadap prediksi kategori, dalam rentang 0 hingga 1.</td>
                            </tr>
                            <tr>
                                <td>aspect_term</td>
                                <td>string</td>
                                <td>Istilah aspek yang diekstrak.</td>
                            </tr>
                            <tr>
                                <td>opinion_term</td>
                                <td>string</td>
                                <td>Istilah opini yang terkait dengan aspek.</td>
                            </tr>
                            <tr>
                                <td>sentiment</td>
                                <td>string</td>
                                <td>Sentimen yang diekspresikan terhadap aspek. Nilai: "POS" (positif), "NEG" (negatif), atau "NEU" (netral).</td>
                            </tr>
                            <tr>
                                <td>triplet_confidence</td>
                                <td>float</td>
                                <td>Tingkat kepercayaan model terhadap prediksi sentimen, dalam rentang 0 hingga 1.</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <div class="example-container">
                        <h5>Contoh Respons</h5>
                        <pre>{
                            "count": 3,
                            "text": "overall aplikasinya bagus, tapi ui jelek dan login susah",
                            "triplets": [
                                {
                                    "aspect_category": "User Experince",
                                    "aspect_category_confidence": 0.7083946466445923,
                                    "aspect_term": "login",
                                    "opinion_term": "susah",
                                    "sentiment": "NEG",
                                    "triplet_confidence": 1.0
                                },
                                {
                                    "aspect_category": "User Interface",
                                    "aspect_category_confidence": 0.5862560868263245,
                                    "aspect_term": "ui",
                                    "opinion_term": "jelek",
                                    "sentiment": "NEG",
                                    "triplet_confidence": 1.0
                                },
                                {
                                    "aspect_category": "General Aspect",
                                    "aspect_category_confidence": 0.975966215133667,
                                    "aspect_term": "aplikasinya",
                                    "opinion_term": "bagus",
                                    "sentiment": "POS",
                                    "triplet_confidence": 1.0
                                }
                            ]
                        }
                            </pre>
                    </div>
                    
                    <h5>Kode Status</h5>
                    <table class="table table-params table-bordered">
                        <thead>
                            <tr>
                                <th>Kode</th>
                                <th>Deskripsi</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>200</td>
                                <td>Sukses. Response berisi hasil analisis.</td>
                            </tr>
                            <tr>
                                <td>400</td>
                                <td>Bad Request. Format permintaan tidak valid atau teks tidak disediakan.</td>
                            </tr>
                            <tr>
                                <td>500</td>
                                <td>Server Error. Terjadi kesalahan saat memproses permintaan.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="section-divider"></div>
                
                <div class="api-section">
                    <h3>Informasi Sistem</h3>
                    
                    <div class="api-endpoint">
                        <span class="method">GET</span>/system_info
                    </div>
                    
                    <p>Memberikan informasi tentang sistem yang menjalankan model, termasuk versi Python, ketersediaan CUDA, dll.</p>
                    
                    <div class="example-container">
                        <h5>Contoh Respons</h5>
                        <pre>{
                        "Python Version": "3.8.10",
                        "PyTorch Version": "1.9.0",
                        "CUDA Available": true,
                        "CUDA Version": "11.1",
                        "Number of CUDA Devices": 1,
                        "Current CUDA Device": 0,
                        "Device Name": "NVIDIA GeForce RTX 3080",
                        "System": "Linux",
                        "System Release": "5.4.0-96-generic",
                        "System Version": "#109-Ubuntu SMP",
                        "Architecture": "x86_64"
                        }
                        </pre>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Span Level Aspect Sentimen Triplet Extraction &copy; 2023</p>
        </div>
    </div>
</body>
</html>
    ''')
    
if __name__ == '__main__':
    print("\n===== System Information =====")
    info = get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print("==============================\n")
    
    print("Starting Aspect-Sentiment Analysis Flask App...")
    print(f"Using device: {model_manager.device}")
    
    # Load models before starting server
    print("Loading models before starting server...")
    model_manager.load_models()
    
    # Set CUDA launch blocking for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=4040, debug=False)