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
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

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

    tic_train = time.time()
    
    training_loss_list = []   # untuk menyimpan loss setiap batch
    valid_steps = []          # untuk menyimpan global step setiap validasi
    valid_f1_list = []        # untuk menyimpan nilai F1 dari validasi
    valid_loss_list = []      # untuk menyimpan nilai loss dari validasi

    global_step = 0
    best_f1 = 0
    loss_list = []
    
    print("\n=== Starting Training ===\n")
    
    for epoch in range(1, args.num_epochs + 1):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            bar_format='{l_bar}{bar:30}{r_bar}',
            file=sys.stdout
        )
        
        model.train()
        for batch_ix, batch in enumerate(progress_bar):
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
            optimizer.step()
            scheduler.step()

            # Simpan nilai loss per batch
            batch_loss = float(loss)
            loss_list.append(batch_loss)
            training_loss_list.append(batch_loss)
            global_step += 1

            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

            # Lakukan validasi setiap valid_steps
            if global_step % args.valid_steps == 0:
                print("\n" + "="*50)
                print(f"Validation at global step {global_step}")
                
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
                
                # Simpan global step dan F1 validasi
                valid_steps.append(global_step)
                valid_f1_list.append(f1)

                if f1 > best_f1:
                    print(f"\n✨ Best F1 score improved: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                
                # Kembali ke mode training
                model.train()
                
                print("="*50 + "\n")

        # Menghitung rata-rata loss per epoch
        avg_loss = sum(loss_list[-len(train_dataloader):]) / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.5f}\n")
    
    # Visualisasi hasil training
    plot_training_results(training_loss_list, valid_steps, valid_f1_list, valid_loss_list)

def plot_training_results(training_loss, valid_steps, valid_f1, valid_loss):
    """
    Plot training loss, validation loss dan f1 score
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(training_loss) + 1), training_loss, 'b-', alpha=0.5, label='Training Loss')
    
    # Plot validation loss pada steps tertentu
    plt.plot(valid_steps, valid_loss, 'r-', marker='o', label='Validation Loss')
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot F1 score
    plt.subplot(2, 1, 2)
    plt.plot(valid_steps, valid_f1, 'g-', marker='o', label='Validation F1 Score')
    
    plt.xlabel('Global Step')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score over Global Steps')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Simpan plot ke file
    plt.tight_layout()
    plt.savefig('training_visualization.png')
    plt.close()
    
    print("Plot saved as 'training_visualization.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="BERT model")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--train_path", default="data/15res", type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default="data/15res", type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=30, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=50, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument("--init_from_ckpt", default=None, type=str,
                        help="The path of model parameters for initialization.")

    args = parser.parse_args()

    do_train()