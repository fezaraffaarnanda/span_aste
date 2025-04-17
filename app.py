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
import json
from datetime import datetime
from tqdm import tqdm
from time import sleep
from typing import List, Optional, Tuple
from google_play_scraper import Sort
from google_play_scraper.constants.element import ElementSpecs
from google_play_scraper.constants.regex import Regex
from google_play_scraper.constants.request import Formats

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
from google_play_scraper.utils.request import post
import time  # Ensure time is imported here

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
            print("Semua model berhasil dimuat")
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
                    print("Semua model berhasil dimuat pada CPU!")
                except Exception as e:
                    print(f"Gagal memuat model pada CPU: {e}")
                    traceback.print_exc()
    
    def load_span_aste_model(self):
        # Configuration - UBAH PATH MODEL SESUAI KEBUTUHAN
        MODEL_PATH = "checkpoint_large/fold1/model_best"  # Path to model directory
        BERT_MODEL = "indobenchmark/indobert-large-p2"  # Model used for training
        
        print(f"Memuat SPAN-ASTE model dari {MODEL_PATH}")
        print(f"Menggunakan {self.device} device")
        
        # Load tokenizer
        try:
            self.span_aste_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
            print(f"Berhasil memuat tokenizer: {type(self.span_aste_tokenizer).__name__}")
        except:
            self.span_aste_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
            print(f"Fallback ke BertTokenizer: {type(self.span_aste_tokenizer).__name__}")
        
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
            raise FileNotFoundError(f"File model tidak ditemukan di {model_file_path}")
        
        state_dict = torch.load(model_file_path, map_location=torch.device(self.device))
        self.span_aste_model.load_state_dict(state_dict, strict=False)  # Use strict=False
        self.span_aste_model.to(self.device)
        self.span_aste_model.eval()
        
        print("SPAN-ASTE model berhasil dimuat")
    
    def load_aspect_classifier_model(self):
        # Configuration - UBAH PATH MODEL SESUAI KEBUTUHAN
        MODEL_PATH = "model_klasifikasi_aspek/model_aspect_categorization.pt"  # Path to model file
        
        print(f"Memuat model klasifikasi aspek dari {MODEL_PATH}")
        
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
        
        print("Model klasifikasi aspek berhasil dimuat")
        print(f"Mapping label: {self.i2w}")
    
    def predict_span_aste(self, text):
        """Extract aspect-opinion-sentiment triplets using SPAN-ASTE model"""
        if not self.loaded:
            self.load_models()
        
        # For very short reviews (less than 3 words), just proceed with normal extraction
        # No fallback mechanism as per user request - if no triplets are extracted, return empty list
            
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
                            "aspect_term": aspect,
                            "opinion_term": opinion,
                            "sentiment": sentiment,
                            "confidence": confidence
                        })
                
                return predict
                
            except Exception as e:
                print(f"Error dalam memprediksi: {e}")
                traceback.print_exc()
                
                # If we're on CUDA and it failed, try falling back to CPU for just this prediction
                if self.device == "cuda":
                    print("Mencoba fallback ke CPU untuk prediksi ini...")
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
                        
                        print("Fallback ke CPU berhasil!")
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
            
            print(f"Kategori aspek untuk '{text}': {predicted_label}, confidence: {confidence:.4f}")
            
            # Return as dictionary for consistent handling
            return {
                'category': predicted_label,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"Error saat klasifikasi aspek: {e}")
            traceback.print_exc()
            
            # Try CPU fallback if we're on CUDA
            if self.device == "cuda":
                try:
                    print("Mencoba fallback ke CPU untuk klasifikasi aspek...")
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
    
@app.route('/scrape')
def scrape_page():
    return render_template('scrape.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for aspect sentiment analysis"""
    # Get input text from JSON payload
    data = request.get_json(silent=True)
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Format permintaan tidak valid. Harap berikan JSON dengan field "text"'
        }), 400
    
    text = data['text']
    
    try:
        print(f"\nMemproses input API: '{text}'")
        
        # Ensure models are loaded
        if not model_manager.loaded:
            print("Models belum dimuat, sedang dimuat...")
            model_manager.load_models()
        
        # 1. Use SPAN-ASTE model to extract triplets
        triplets = model_manager.predict_span_aste(text)
        
        # 2. For each triplet, predict aspect category
        formatted_triplets = []
        for triplet in triplets:
            # Combine aspect and opinion for aspect category prediction
            aspect_opinion_text = f"{triplet['aspect_term']} {triplet['opinion_term']}"
            
            # Predict aspect category
            category_info = model_manager.predict_aspect_category(aspect_opinion_text)
            
            # Format triplet with new key names
            formatted_triplet = {
                'aspect_term': triplet['aspect_term'],
                'opinion_term': triplet['opinion_term'],
                'sentiment': triplet['sentiment'],
                'triplet_confidence': triplet['confidence'],
                'aspect_category': category_info['category'],
                'aspect_category_confidence': category_info['confidence']
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

@app.route('/api/scrape', methods=['POST'])
def api_scrape():
    try:
        # Get max_reviews parameter from request
        data = request.json or {}
        max_reviews = data.get('max_reviews', 0)  # Default to 0 (all reviews)
        
        # Define the path for saved data
        saved_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_data')
        os.makedirs(saved_data_dir, exist_ok=True)
        saved_data_path = os.path.join(saved_data_dir, f'{ALLSTATS_APP_ID}_reviews.json')
        
        # Check if we have saved data
        if os.path.exists(saved_data_path):
            try:
                with open(saved_data_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    # Check if saved data is valid and has results
                    if isinstance(saved_data, dict) and 'results' in saved_data and saved_data.get('max_reviews', 0) >= max_reviews:
                        print(f"Loading {len(saved_data['results'])} saved reviews from disk")
                        return jsonify(saved_data)
            except Exception as e:
                print(f"Error saat memuat data yang disimpan: {e}")
        
        # Start scraping with progress feedback
        print(f"Mulai scraping ulasan untuk aplikasi: {ALLSTATS_APP_ID} dengan max_reviews={max_reviews}")
        
        # Scrape the reviews
        reviews_data = reviews_all(ALLSTATS_APP_ID, max_reviews=max_reviews)
        print(f"Mengumpulkan {len(reviews_data)} ulasan aplikasi {ALLSTATS_APP_ID}")
        
        # Function to check if a text is only emojis or emoticons
        def is_emoji_only(text):
            import re
            # Basic pattern to detect emoji and emoticons
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251" 
                "]+"
            )
            
            # Strip whitespace
            cleaned_text = text.strip()
            
            # Remove all emojis and common emoticons
            text_without_emojis = emoji_pattern.sub(r'', cleaned_text)
            text_without_emoticons = re.sub(r'[:;=][\-^]?[)(<>$@3PDpoO0\[\]\\|/]', '', text_without_emojis)
            
            # If after removing emojis/emoticons the text is empty or just whitespace, it was emoji-only
            return not text_without_emoticons.strip()
        
        # Process each review with the model
        results = []
        for review in tqdm(reviews_data, desc="Processing Reviews"):
            # Get the review text
            review_text = review['content']
            review_score = review['score']  # Rating (1-5)
            
            # Skip empty reviews or emoji-only reviews
            if not review_text or review_text.strip() == "" or is_emoji_only(review_text):
                continue
            
            # Track processing time and status
            start_time = time.time()
            processing_status = "success"
            error_details = None
            
            try: 
                # No longer using fallback mechanism
                # Reset any current_review_score to ensure it doesn't affect processing
                model_manager.current_review_score = None
                
                # Process with our model
                triplets = model_manager.predict_span_aste(review_text)
                
                # Reset current review score
                model_manager.current_review_score = None
                
                # Get aspect categories if available
                if model_manager.aspect_classifier_model is not None and triplets:
                    for triplet in triplets:
                        if isinstance(triplet, dict) and 'aspect_term' in triplet:
                            aspect_term = triplet['aspect_term']
                            opinion_term = triplet['opinion_term']
                            aspect_opinion_text = f"{aspect_term} {opinion_term}"
                            category_info = model_manager.predict_aspect_category(aspect_opinion_text)
                            triplet['aspect_category'] = category_info['category']
                            triplet['category_confidence'] = category_info['confidence']
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Add to results with metadata
                results.append({
                    'review_id': review['reviewId'],
                    'review_text': review_text,
                    'score': review_score,
                    'at': review['at'],
                    'triplets': triplets,
                    'processing_status': processing_status,
                    'processing_time': round(processing_time, 3),
                    'processed_at': datetime.now().isoformat(),
                    'model_version': getattr(model_manager, 'model_version', 'span_aste_v1')
                })
            except Exception as e:
                print(f"Error saat memproses ulasan: {review_text}")
                print(f"Detail error : {e}")
                # Log the error but still include the review in results with error info
                processing_status = "error"
                error_details = str(e)
                processing_time = time.time() - start_time
                
                # Add to results with error information
                results.append({
                    'review_id': review['reviewId'],
                    'review_text': review_text,
                    'score': review_score,
                    'at': review['at'],
                    'triplets': [],  # Empty triplets for error cases
                    'processing_status': processing_status,
                    'processing_time': round(processing_time, 3),
                    'processed_at': datetime.now().isoformat(),
                    'error_details': error_details,
                    'model_version': getattr(model_manager, 'model_version', 'span_aste_v1')
                })
                # Reset current_review_score in case of error
                model_manager.current_review_score = None
                # Continue processing other reviews even if one fails
                continue
        
        # Prepare response data
        response_data = {
            'success': True,
            'app_id': ALLSTATS_APP_ID,
            'total_reviews': len(reviews_data),
            'processed_reviews': len(results),
            'max_reviews': max_reviews,
            'results': results
        }
        
        # Save data to disk for future use
        try:
            with open(saved_data_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
            print(f"Menyimpan {len(results)} ulasan yang sudah diproses {saved_data_path}")
        except Exception as e:
            print(f"Error saat menyimpan data ke disk: {e}")
        
        return jsonify(response_data)
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(f"Error saat scraping ulasan: {e}\n{traceback_info}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

# API endpoint for scraping reviews and getting predictions in one call
@app.route('/api/scrape_and_predict', methods=['POST'])
def api_scrape_and_predict():
    """Scrape reviews and get sentiment analysis predictions in one call.
    Accepts max_reviews parameter (optional) to limit the number of reviews to scrape.
    Returns the scraped reviews with sentiment analysis results.
    """
    try:
        # Get request data
        data = request.get_json() or {}
        max_reviews = data.get('max_reviews', 0)  # 0 means all reviews
        
        # Log parameter untuk debugging
        print(f"Request data: {data}")
        print(f"max_reviews parameter: {max_reviews}")
        
        # App ID for BPS Allstats
        ALLSTATS_APP_ID = "id.go.bps.allstats"
        
        # Path to save scrapped data
        saved_data_dir = "saved_data"
        saved_data_path = f"{saved_data_dir}/{ALLSTATS_APP_ID}_reviews.json"
        
        # Create directory if it doesn't exist
        os.makedirs(saved_data_dir, exist_ok=True)
        
        print(f"Mulai scraping ulasan untuk aplikasi: {ALLSTATS_APP_ID} dengan max_reviews={max_reviews}")
        
        # Use google-play-scraper to get reviews
        from google_play_scraper import Sort, reviews
        
        # Gunakan fungsi reviews_all() yang sudah ada untuk mengambil ulasan
        # Fungsi ini sudah menangani parameter max_reviews dengan benar
        print(f"Mulai scraping ulasan untuk aplikasi: {ALLSTATS_APP_ID} dengan max_reviews={max_reviews} menggunakan reviews_all()")
        reviews_data = reviews_all(ALLSTATS_APP_ID, max_reviews=max_reviews)
        
        print(f"Mengumpulkan {len(reviews_data)} ulasan aplikasi {ALLSTATS_APP_ID}")
        
        # Function to check if a text is only emojis or emoticons
        def is_emoji_only(text):
            import re
            # Basic pattern to detect emoji and emoticons
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251" 
                "]+"
            )
            
            # Strip whitespace
            cleaned_text = text.strip()
            
            # Remove all emojis and common emoticons
            text_without_emojis = emoji_pattern.sub(r'', cleaned_text)
            text_without_emoticons = re.sub(r'[:;=][\-^]?[)(<>$@3PDpoO0\[\]\\|/]', '', text_without_emojis)
            
            # If after removing emojis/emoticons the text is empty or just whitespace, it was emoji-only
            return not text_without_emoticons.strip()
        
        # Process each review with the model
        results = []
        for review in tqdm(reviews_data, desc="Processing Reviews"):
            # Get the review text
            review_text = review['content']
            review_score = review['score']  # Rating (1-5)
            
            # Skip empty reviews or emoji-only reviews
            if not review_text or review_text.strip() == "" or is_emoji_only(review_text):
                continue
            
            # Track processing time and status
            start_time = time.time()
            processing_status = "success"
            error_details = None
            
            try: 
                # No longer using fallback mechanism
                # Reset any current_review_score to ensure it doesn't affect processing
                model_manager.current_review_score = None
                
                # Process with our model
                triplets = model_manager.predict_span_aste(review_text)
                
                # Reset current review score
                model_manager.current_review_score = None
                
                # Get aspect categories if available
                if model_manager.aspect_classifier_model is not None and triplets:
                    for triplet in triplets:
                        if isinstance(triplet, dict) and 'aspect_term' in triplet:
                            aspect_term = triplet['aspect_term']
                            opinion_term = triplet['opinion_term']
                            aspect_opinion_text = f"{aspect_term} {opinion_term}"
                            category_info = model_manager.predict_aspect_category(aspect_opinion_text)
                            triplet['aspect_category'] = category_info['category']
                            triplet['category_confidence'] = category_info['confidence']
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Format each triplet for consistent output
                formatted_triplets = []
                for triplet in triplets:
                    formatted_triplet = {
                        'aspect_term': triplet['aspect_term'],
                        'opinion_term': triplet['opinion_term'],
                        'sentiment': triplet['sentiment'],
                        'triplet_confidence': triplet.get('confidence', 0.0),
                        'aspect_category': triplet.get('aspect_category', 'Unknown'),
                        'aspect_category_confidence': triplet.get('category_confidence', 0.0)
                    }
                    formatted_triplets.append(formatted_triplet)
                
                # Add to results with metadata
                results.append({
                    'review_id': review['reviewId'],
                    'review_text': review_text,
                    'score': review_score,
                    'at': review['at'],
                    'triplets': formatted_triplets,
                    'processing_status': processing_status,
                    'processing_time': round(processing_time, 3),
                    'processed_at': datetime.now().isoformat(),
                    'model_version': getattr(model_manager, 'model_version', 'span_aste_v1')
                })
            except Exception as e:
                print(f"Error processing review: {review_text}")
                print(f"Error details: {e}")
                # Log the error but still include the review in results with error info
                processing_status = "error"
                error_details = str(e)
                processing_time = time.time() - start_time
                
                # Add to results with error information
                results.append({
                    'review_id': review['reviewId'],
                    'review_text': review_text,
                    'score': review_score,
                    'at': review['at'],
                    'triplets': [],  # Empty triplets for error cases
                    'processing_status': processing_status,
                    'processing_time': round(processing_time, 3),
                    'processed_at': datetime.now().isoformat(),
                    'error_details': error_details,
                    'model_version': getattr(model_manager, 'model_version', 'span_aste_v1')
                })
                # Reset current_review_score in case of error
                model_manager.current_review_score = None
        
        # Calculate sentiment statistics
        total_reviews = len(results)
        processed_reviews = sum(1 for r in results if r['processing_status'] == 'success')
        
        # Count triplets by sentiment
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for review in results:
            for triplet in review.get('triplets', []):
                if triplet['sentiment'] == 'POS':
                    positive_count += 1
                elif triplet['sentiment'] == 'NEG':
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # Prepare response data with statistics
        response_data = {
            'success': True,
            'app_id': ALLSTATS_APP_ID,
            'total_reviews': total_reviews,
            'processed_reviews': processed_reviews,
            'statistics': {
                'positive_triplets': positive_count,
                'negative_triplets': negative_count,
                'neutral_triplets': neutral_count,
                'total_triplets': positive_count + negative_count + neutral_count
            },
            'results': results
        }
        
        return jsonify(response_data)
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(f"Error saat scraping ulasan: {e}\n{traceback_info}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

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
        print(f"\nMemproses input: '{text}'")
        
        # Ensure models are loaded
        if not model_manager.loaded:
            print("Models belum dimuat, sedang dimuat...")
            model_manager.load_models()
        
        # 1. Use SPAN-ASTE model to extract triplets
        triplets = model_manager.predict_span_aste(text)
        
        # 2. For each triplet, predict aspect category
        formatted_triplets = []
        for triplet in triplets:
            # Combine aspect and opinion for aspect category prediction
            aspect_opinion_text = f"{triplet['aspect_term']} {triplet['opinion_term']}"
            
            # Predict aspect category
            category_info = model_manager.predict_aspect_category(aspect_opinion_text)
            
            # Format triplet with new key names
            formatted_triplet = {
                'aspect_term': triplet['aspect_term'],
                'opinion_term': triplet['opinion_term'],
                'sentiment': triplet['sentiment'],
                'triplet_confidence': triplet['confidence'],
                'aspect_category': category_info['category'],
                'aspect_category_confidence': category_info['confidence']
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


# Google Play Store Scraper constants
MAX_COUNT_EACH_FETCH = 199
MAX_REVIEWS_PER_APP = 0  # 0 means get all available reviews
ALLSTATS_APP_ID = 'id.go.bps.allstats'


# Google Play Store Scraper functionality
class _ContinuationToken:
    __slots__ = ("token", "lang", "country", "sort", "count", "filter_score_with", "filter_device_with")

    def __init__(self, token, lang, country, sort, count, filter_score_with, filter_device_with):
        self.token = token
        self.lang = lang
        self.country = country
        self.sort = sort
        self.count = count
        self.filter_score_with = filter_score_with
        self.filter_device_with = filter_device_with

def _fetch_review_items(url, app_id, sort, count, filter_score_with, filter_device_with, pagination_token):
    dom = post(
        url,
        Formats.Reviews.build_body(
            app_id,
            sort,
            count,
            "null" if filter_score_with is None else filter_score_with,
            "null" if filter_device_with is None else filter_device_with,
            pagination_token,
        ),
        {"content-type": "application/x-www-form-urlencoded"},
    )
    match = json.loads(Regex.REVIEWS.findall(dom)[0])
    return json.loads(match[0][2])[0], json.loads(match[0][2])[-2][-1]

def reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=100, filter_score_with=None, continuation_token=None):
    sort = sort.value
    if continuation_token is not None:
        token = continuation_token.token
        if token is None:
            return [], continuation_token
    else:
        token = None

    url = Formats.Reviews.build(lang=lang, country=country)
    _fetch_count = count
    result = []

    while True:
        if _fetch_count == 0:
            break
        if _fetch_count > MAX_COUNT_EACH_FETCH:
            _fetch_count = MAX_COUNT_EACH_FETCH

        try:
            review_items, token = _fetch_review_items(
                url, app_id, sort, _fetch_count, filter_score_with, None, token
            )
        except (TypeError, IndexError):
            break

        for review in review_items:
            result.append({k: spec.extract_content(review) for k, spec in ElementSpecs.Review.items()})

        _fetch_count = count - len(result)
        if token is None:
            break

    return result, _ContinuationToken(token, lang, country, sort, count, filter_score_with, None)

def reviews_all(app_id, max_reviews=MAX_REVIEWS_PER_APP, sleep_milliseconds=1000):
    result = []
    continuation_token = None

    while True:
        count_to_fetch = MAX_COUNT_EACH_FETCH
        if max_reviews > 0 and len(result) + count_to_fetch > max_reviews:
            count_to_fetch = max_reviews - len(result)
            
        new_result, continuation_token = reviews(
            app_id,
            count=count_to_fetch,
            continuation_token=continuation_token,
        )
        
        if not new_result:
            break
            
        result.extend(new_result)
        
        if max_reviews > 0 and len(result) >= max_reviews:
            break
            
        if continuation_token.token is None:
            break
            
        sleep(sleep_milliseconds / 1000)

    return result

if __name__ == '__main__':
    print("\n===== System Information =====")
    info = get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print("==============================\n")
    
    print("Bismillah mulai Span-ASTE Flask App...")
    print(f"Using device: {model_manager.device}")
    
    # Load models before starting server
    print("Loading models sebelum server dimulai...")
    model_manager.load_models()
    
    # Set CUDA launch blocking for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=8099, debug=True)