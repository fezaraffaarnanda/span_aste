import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

from evaluate import evaluate
from models.losses import log_likelihood
from models.metrics import SpanEvaluator
from utils.bar import ProgressBar
from utils.dataset import CustomDataset
from models.collate import collate_fn, gold_labels
import numpy as np
from models.model import SpanAsteModel
from utils.processor import Res15DataProcessor
from utils.tager import SpanLabel
from utils.tager import RelationLabel
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"using device:{device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def do_train():
    # set seed
    set_seed(args.seed)

    # Track overall training time
    training_start_time = time.time()

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # create processor
    processor = Res15DataProcessor(tokenizer, args.max_seq_len)

    print("Loading Train & Eval Dataset...")
    # Load dataset
    train_dataset = CustomDataset("train", args.train_path, processor, tokenizer, args.max_seq_len)
    eval_dataset = CustomDataset("dev", args.dev_path, processor, tokenizer, args.max_seq_len)

    print("Construct Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("Building SPAN-ASTE model...")
    # get dimension of target and relation
    target_dim, relation_dim = len(SpanLabel), len(RelationLabel)
    # build span-aste model
    model = SpanAsteModel(
        args.bert_model,
        target_dim,
        relation_dim,
        device=device
    )
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    bert_param_optimizer = list(model.bert.named_parameters())
    span_linear_param_optimizer = list(model.span_ffnn.named_parameters())
    pair_linear_param_optimizer = list(model.pairs_ffnn.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in span_linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-3},
        {'params': [p for n, p in span_linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 1e-3},

        {'params': [p for n, p in pair_linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-3},
        {'params': [p for n, p in pair_linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 1e-3}
    ]

    print("Building Optimizer...")
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = num_training_steps * args.warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    metric = SpanEvaluator()
    
    training_loss_list = []   # untuk menyimpan loss setiap batch
    epoch_list = []           # untuk menyimpan nomor epoch untuk visualisasi
    valid_f1_list = []        # untuk menyimpan nilai F1 dari validasi
    valid_loss_list = []      # untuk menyimpan nilai loss dari validasi
    epoch_avg_loss_list = []  # untuk menyimpan rata-rata loss per epoch
    
    # Untuk menyimpan data waktu
    epoch_durations = []      # durasi setiap epoch dalam detik
    batch_times = []          # timestamp untuk setiap batch
    validation_durations = [] # durasi validasi setiap epoch

    global_step = 0
    best_f1 = 0
    
    print("\n=== Starting Fine Tuning ===\n")
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            bar_format='{l_bar}{bar:30}{r_bar}',
            file=sys.stdout
        )
        
        # Untuk menyimpan loss per epoch
        epoch_losses = []
        
        model.train()
        for batch_ix, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Proses forward dan backward seperti biasa
            input_ids, attention_mask, token_type_ids, spans, relations, span_labels, relation_labels, seq_len = batch
            input_ids = torch.tensor(input_ids, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            token_type_ids = torch.tensor(token_type_ids, device=device)

            # forward
            spans_probability, span_indices, relations_probability, candidate_indices = model(
                input_ids, attention_mask, token_type_ids, seq_len)

            gold_span_indices, gold_span_labels = gold_labels(span_indices, spans, span_labels)
            loss_ner = log_likelihood(spans_probability, span_indices, gold_span_indices, gold_span_labels)

            gold_relation_indices, gold_relation_labels = gold_labels(candidate_indices, relations, relation_labels)
            loss_relation = log_likelihood(relations_probability, candidate_indices, gold_relation_indices, gold_relation_labels)

            # Kombinasi loss
            loss = 0.2 * loss_ner + loss_relation

            # backward
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping untuk mencegah gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Tambahkan ini
            optimizer.step()
            scheduler.step()

            # Simpan nilai loss per batch
            batch_loss = float(loss)
            training_loss_list.append(batch_loss)
            epoch_losses.append(batch_loss)
            global_step += 1
            
            # Catat waktu batch
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            batch_times.append({
                'epoch': epoch,
                'batch': batch_ix + 1,
                'global_step': global_step,
                'timestamp': batch_end_time,
                'duration': batch_duration
            })

            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'time': f'{batch_duration:.2f}s'})

        # Menghitung rata-rata loss per epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_loss_list.append(avg_loss)
        epoch_time = time.time() - epoch_start_time
        epoch_durations.append(epoch_time)
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s. Average loss: {avg_loss:.5f}")
        
        # Validasi di akhir setiap epoch
        print("\n" + "="*50)
        print(f"Validation at end of epoch {epoch}")
        
        # Start validation time measurement
        validation_start_time = time.time()
        
        # Hitung validation loss
        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for valid_batch in eval_dataloader:
                v_input_ids, v_attention_mask, v_token_type_ids, v_spans, v_relations, v_span_labels, v_relation_labels, v_seq_len = valid_batch
                v_input_ids = torch.tensor(v_input_ids, device=device)
                v_attention_mask = torch.tensor(v_attention_mask, device=device)
                v_token_type_ids = torch.tensor(v_token_type_ids, device=device)

                # forward
                v_spans_probability, v_span_indices, v_relations_probability, v_candidate_indices = model(
                    v_input_ids, v_attention_mask, v_token_type_ids, v_seq_len)

                v_gold_span_indices, v_gold_span_labels = gold_labels(v_span_indices, v_spans, v_span_labels)
                v_loss_ner = log_likelihood(v_spans_probability, v_span_indices, v_gold_span_indices, v_gold_span_labels)

                v_gold_relation_indices, v_gold_relation_labels = gold_labels(v_candidate_indices, v_relations, v_relation_labels)
                v_loss_relation = log_likelihood(v_relations_probability, v_candidate_indices, v_gold_relation_indices, v_gold_relation_labels)

                # Kombinasi loss
                v_loss = 0.2 * v_loss_ner + v_loss_relation
                valid_loss += float(v_loss)
                valid_batches += 1
        
        # Calculate validation duration
        validation_duration = time.time() - validation_start_time
        validation_durations.append(validation_duration)
        
        avg_valid_loss = valid_loss / valid_batches if valid_batches > 0 else 0
        valid_loss_list.append(avg_valid_loss)
        
        # Evaluasi metric F1
        precision, recall, f1 = evaluate(model, metric, eval_dataloader, device)
        print(
            "Evaluation metrics:\n"
            f"  Precision: {precision:.5f}\n"
            f"  Recall: {recall:.5f}\n"
            f"  F1: {f1:.5f}\n"
            f"  Validation Loss: {avg_valid_loss:.5f}"
        )
        
        # Simpan epoch number dan F1 validasi
        epoch_list.append(epoch)
        valid_f1_list.append(f1)

        if f1 > best_f1:
            print(f"\n✨ Best F1 score improved: {best_f1:.5f} --> {f1:.5f}")
            best_f1 = f1
            save_dir = os.path.join(args.save_dir, "model_best")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            
            # Optionally save model for specific epoch if args.save_all_epochs is True
            if args.save_all_epochs:
                epoch_save_dir = os.path.join(args.save_dir, f"model_epoch_{epoch}")
                if not os.path.exists(epoch_save_dir):
                    os.makedirs(epoch_save_dir)
                torch.save(model.state_dict(), os.path.join(epoch_save_dir, "model.pt"))
        
        # Kembali ke mode training
        model.train()
        
        print("="*50 + "\n")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    
    # Visualisasi hasil training
    if args.visualize:
        plot_training_results(
            training_loss_list, epoch_list, valid_f1_list, valid_loss_list, 
            epoch_avg_loss_list, epoch_durations, validation_durations, batch_times, total_training_time
        )

def plot_training_results(training_loss, epoch_list, valid_f1, valid_loss, epoch_avg_loss, 
                       epoch_durations, validation_durations, batch_times, total_training_time):
    """
    Plot training loss, validation loss dan f1 score berdasarkan epoch
    """
    # Create visualization directory if it doesn't exist
    if not os.path.exists(args.viz_dir):
        os.makedirs(args.viz_dir)
    
    # Construct full paths for visualization files
    overview_plot_path = os.path.join(args.viz_dir, args.viz_filename)
    combined_metrics_path = os.path.join(args.viz_dir, f"combined_metrics_{args.viz_filename}")
    time_metrics_path = os.path.join(args.viz_dir, f"time_metrics_{args.viz_filename}")
    
    # Plot training loss, validation loss and F1 score
    plt.figure(figsize=(15, 15))
    
    # Plot training loss per batch
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(training_loss) + 1), training_loss, 'b-', alpha=0.5, label='Training Loss (per batch)')
    
    # Menambahkan marker untuk epoch boundaries
    batch_per_epoch = len(training_loss) // len(epoch_list)
    epoch_boundaries = [i * batch_per_epoch for i in range(1, len(epoch_list))]
    plt.vlines(epoch_boundaries, min(training_loss), max(training_loss), colors='gray', linestyles='dashed', alpha=0.3)
    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training & validation loss per epoch
    plt.subplot(3, 1, 2)
    plt.plot(epoch_list, epoch_avg_loss, 'b-', marker='o', label='Training Loss per Epoch')
    plt.plot(epoch_list, valid_loss, 'r-', marker='o', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xticks(epoch_list)  # Memastikan sumbu x menampilkan epoch secara eksplisit
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot F1 score per epoch
    plt.subplot(3, 1, 3)
    plt.plot(epoch_list, valid_f1, 'g-', marker='o', label='Validation F1 Score')
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score per Epoch')
    plt.xticks(epoch_list)  # Memastikan sumbu x menampilkan epoch secara eksplisit
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Simpan plot ke file
    plt.tight_layout()
    plt.savefig(overview_plot_path)
    plt.close()
    
    # Buat plot tambahan yang menggabungkan validation loss dan F1 score dalam satu grafik
    plt.figure(figsize=(10, 6))
    
    # Plot untuk validation loss (sumbu y kiri)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color='r')
    ax1.plot(epoch_list, valid_loss, 'r-', marker='o', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xticks(epoch_list)
    
    # Plot untuk F1 score (sumbu y kanan)
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='g')
    ax2.plot(epoch_list, valid_f1, 'g-', marker='o', label='F1 Score')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title('Validation Metrics per Epoch')
    
    # Menambahkan legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(combined_metrics_path)
    plt.close()
    
    # Plot time metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot epoch durations
    ax1.bar(epoch_list, epoch_durations, color='teal', alpha=0.7)
    ax1.plot(epoch_list, validation_durations, 'ro-', label='Validation Duration')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Training and Validation Duration per Epoch')
    ax1.set_xticks(epoch_list)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cumulative training time
    cumulative_time = np.cumsum(epoch_durations)
    ax2.plot(epoch_list, cumulative_time, 'mo-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cumulative Time (seconds)')
    ax2.set_title('Cumulative Training Time')
    ax2.set_xticks(epoch_list)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation for total training time
    ax2.annotate(f'Total: {total_training_time:.1f}s ({total_training_time/60:.1f}min)', 
                xy=(epoch_list[-1], cumulative_time[-1]),
                xytext=(-150, 30), 
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    plt.savefig(time_metrics_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="BERT model")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay rate for AdamW.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training steps for learning rate warmup.")
    parser.add_argument("--train_path", default="data/15res", type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default="data/15res", type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--init_from_ckpt", default=None, type=str,
                        help="The path of model parameters for initialization.")

    # Visualization parameters
    parser.add_argument("--visualize", action="store_true", 
                        help="Whether to generate visualization plots")
    parser.add_argument("--viz_dir", default="./visualizations", type=str,
                        help="Directory to save visualization results")
    parser.add_argument("--viz_filename", default="training_visualization.png", type=str,
                        help="Filename for the visualization plot")
    parser.add_argument("--save_all_epochs", action="store_true", 
                        help="Whether to save model checkpoints for all epochs")
    parser.add_argument("--export_excel", action="store_true", default=True,
                        help="Export metrics as Excel file with multiple sheets")
    parser.add_argument("--excel_filename", default="training_metrics.xlsx", type=str,
                        help="Filename for the Excel metrics")

    args = parser.parse_args()

    do_train()