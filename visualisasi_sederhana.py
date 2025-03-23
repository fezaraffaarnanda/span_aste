#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Panduan Lengkap Fine-tuning Model SPAN-ASTE untuk Pemula
========================================================
File ini menjelaskan proses fine-tuning model SPAN-ASTE langkah demi langkah 
dengan bahasa yang sederhana untuk pemahaman yang lebih baik.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# -----------------
# BAGIAN 1: MEMAHAMI DATA INPUT
# -----------------
print(" MEMAHAMI DATA INPUT SPAN-ASTE")
print("=" * 70)

print("""
Model SPAN-ASTE digunakan untuk mengekstrak "triplet sentimen" dari kalimat, yaitu:
1. ASPEK - apa yang dibicarakan? (misalnya: "aplikasi", "login", "server")
2. OPINI - bagaimana pendapat tentang aspek tersebut? (misalnya: "sangat bagus", "tidak bisa")
3. SENTIMEN - apakah opini tersebut positif, negatif, atau netral?

Format data input model adalah: "teks####[([indeks_aspek], [indeks_opini], 'sentimen')]"
""")

# Contoh data
data_contoh = [
    # Contoh 1
    {
        "teks": "aplikasi sangat bagus membantu pegawai meningkatkan kinernya.",
        "label": [([0], [1, 2], 'POS')],
        "penjelasan": """
        >>> Teks: "aplikasi sangat bagus membantu pegawai meningkatkan kinernya."
        >>> Triplet Sentimen:
            - ASPEK: "aplikasi" (kata ke-0)
            - OPINI: "sangat bagus" (kata ke-1 dan ke-2)
            - SENTIMEN: Positif
        """
    },
    # Contoh 2
    {
        "teks": "gak bisa login tulisannya sedang dalam pemeliharaan.",
        "label": [([2], [0, 1], 'NEG')],
        "penjelasan": """
        >>> Teks: "gak bisa login tulisannya sedang dalam pemeliharaan."
        >>> Triplet Sentimen:
            - ASPEK: "login" (kata ke-2)
            - OPINI: "gak bisa" (kata ke-0 dan ke-1)
            - SENTIMEN: Negatif
        """
    }
]

# Menampilkan contoh data
for i, contoh in enumerate(data_contoh):
    print(f"\n CONTOH DATA #{i+1}:")
    print(f"Teks: \"{contoh['teks']}\"")
    print(f"Label: {contoh['label']}")
    print(contoh['penjelasan'])

# -----------------
# BAGIAN 2: PERSIAPAN DATA (PREPROCESSING)
# -----------------
print("\n\n PERSIAPAN DATA UNTUK FINE-TUNING")
print("=" * 70)

print("""
Sebelum model bisa belajar, teks perlu dikonversi ke format yang dipahami model.
Ini seperti menerjemahkan bahasa manusia ke bahasa komputer.
""")

contoh_teks = data_contoh[0]['teks']
print(f"Teks contoh: \"{contoh_teks}\"")

# Inisialisasi tokenizer
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")

# Memecah teks menjadi kata-kata
kata_kata = contoh_teks.split()
print(f"\n1️⃣ Teks dipecah menjadi kata-kata:")
for i, kata in enumerate(kata_kata):
    print(f"   Kata #{i}: '{kata}'")

# Tokenisasi BERT
token_bert = tokenizer.tokenize(contoh_teks)
print(f"\n2️⃣ Kata-kata dipecah menjadi token BERT:")
print(f"   Token BERT: {token_bert}")

# Mapping kata ke token
token_idx = 0
print(f"\n3️⃣ Pemetaan kata ke token BERT:")
for word_idx, word in enumerate(kata_kata):
    word_tokens = tokenizer.tokenize(word)
    print(f"   Kata[{word_idx}]: '{word}' → Token: {word_tokens}")
    token_idx += len(word_tokens)

# Konversi token ke angka (IDs)
inputs = tokenizer.encode_plus(
    contoh_teks, 
    max_length=128, 
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)

print(f"\n4️⃣ Token dikonversi ke angka (IDs) yang dipahami model:")
print(f"   Input IDs: {inputs.input_ids[0][:20]}...")
print(f"   Attention Mask: {inputs.attention_mask[0][:20]}...")

# -----------------
# BAGIAN 3: KONVERSI INDEKS KATA KE INDEKS TOKEN
# -----------------
print("\n5️⃣ Konversi indeks ASPEK dan OPINI dari indeks kata ke indeks token:")

# Contoh indeks
aspek_idx = data_contoh[0]['label'][0][0]  # [0]
opini_idx = data_contoh[0]['label'][0][1]  # [1, 2]

print(f"   Indeks kata ASPEK: {aspek_idx} → '{' '.join([kata_kata[i] for i in aspek_idx])}'")
print(f"   Indeks kata OPINI: {opini_idx} → '{' '.join([kata_kata[i] for i in opini_idx])}'")

# Hitung token offset (posisi dalam token BERT)
def hitung_offset_token(kata_sebelum, kata_target, tokenizer):
    """Menghitung posisi token dalam tokenisasi BERT"""
    # Token untuk teks sebelum target
    token_sebelum = len(tokenizer.encode(kata_sebelum, add_special_tokens=True)) - 1
    
    # Panjang token target
    token_target = len(tokenizer.tokenize(kata_target))
    
    # Offset
    start_offset = token_sebelum
    end_offset = start_offset + token_target - 1
    
    return start_offset, end_offset

# Untuk aspek
kata_sebelum_aspek = " ".join(kata_kata[:aspek_idx[0]])
kata_aspek = kata_kata[aspek_idx[0]]
aspek_start, aspek_end = hitung_offset_token(kata_sebelum_aspek, kata_aspek, tokenizer)

# Untuk opini
kata_sebelum_opini = " ".join(kata_kata[:opini_idx[0]])
kata_opini = " ".join([kata_kata[i] for i in opini_idx])
opini_start, opini_end = hitung_offset_token(kata_sebelum_opini, kata_opini, tokenizer)

print(f"   Token BERT untuk ASPEK: posisi [{aspek_start}, {aspek_end}]")
print(f"   Token BERT untuk OPINI: posisi [{opini_start}, {opini_end}]")

# -----------------
# BAGIAN 4: MEMAHAMI MODEL SPAN-ASTE
# -----------------
print("\n\n易 MODEL SPAN-ASTE: BAGAIMANA INI BEKERJA?")
print("=" * 70)

print("""
Model SPAN-ASTE memiliki cara kerja yang mirip dengan seseorang yang:
1. Membaca kalimat dan memahami artinya (via BERT)
2. Menandai bagian-bagian teks yang mungkin aspek atau opini
3. Memilih pasangan aspek-opini yang masuk akal
4. Menentukan sentimen dari tiap pasangan aspek-opini

Mari kita bahas langkah demi langkah:
""")

# Ilustrasi Model
print("\n1️⃣ PEMAHAMAN TEKS DENGAN BERT")
print("""
   BERT adalah 'otak' model yang memahami konteks kalimat.
   
   Ilustrasi: 
   "Gak bisa login" → BERT → [0.2, -0.3, 0.7...] (vektor representasi)
   
   Setiap kata mendapatkan vektor yang merepresentasikan artinya dalam konteks.
""")

print("\n2️⃣ PEMBENTUKAN SPAN (RENTANG TEKS)")
print("""
   Span adalah rentang teks berurutan, misalnya:
   Untuk kalimat: "aplikasi sangat bagus"
   
   Semua kemungkinan span:
   - (0,0): "aplikasi"
   - (1,1): "sangat"
   - (2,2): "bagus"
   - (0,1): "aplikasi sangat"
   - (1,2): "sangat bagus"
   - (0,2): "aplikasi sangat bagus"
   
   Setiap span direpresentasikan dengan:
   span_i,j = [vektor_token_awal; vektor_token_akhir; embedding_panjang_span]
""")

print("\n3️⃣ KLASIFIKASI SPAN")
print("""
   Setiap span diklasifikasikan sebagai:
   - INVALID (0): Bukan aspek atau opini
   - ASPECT (1): Span adalah aspek
   - OPINION (2): Span adalah opini
   
   Misalnya:
   "aplikasi" → Aspek (skor tinggi)
   "sangat bagus" → Opini (skor tinggi)
   "aplikasi sangat" → Invalid (skor rendah)
""")

print("\n4️⃣ PRUNING (PEMANGKASAN SPAN)")
print("""
   Karena jumlah span bisa sangat banyak, kita pilih hanya top-K span.
   
   Misalnya, dari 100 kemungkinan span:
   - Simpan 10 span dengan skor ASPECT tertinggi
   - Simpan 10 span dengan skor OPINION tertinggi
""")

print("\n5️⃣ PEMBENTUKAN PASANGAN TARGET-OPINI")
print("""
   Setiap span aspek dipasangkan dengan setiap span opini.
   
   Representasi pasangan:
   pair_a,b,c,d = [span_aspek; span_opini; embedding_jarak]
   
   di mana embedding_jarak merepresentasikan seberapa dekat aspek dan opini.
""")

print("\n6️⃣ KLASIFIKASI RELASI")
print("""
   Setiap pasangan diklasifikasikan menjadi:
   - INVALID (0): Tidak ada relasi
   - POS (1): Sentimen positif
   - NEG (2): Sentimen negatif
   - NEU (3): Sentimen netral
   
   Output akhir: triplet (aspek, opini, sentimen)
   Misalnya: ("aplikasi", "sangat bagus", "POS")
""")

# -----------------
# BAGIAN 5: PROSES FINE-TUNING LANGKAH-DEMI-LANGKAH
# -----------------
print("\n\n PROSES FINE-TUNING MODEL SPAN-ASTE")
print("=" * 70)

print("""
Fine-tuning adalah proses "mengajari" model untuk tugas spesifik.
Ini mirip dengan mengajari seseorang yang sudah memiliki pengetahuan umum
(BERT pre-trained) untuk melakukan tugas spesifik (ekstraksi triplet sentimen).

Mari kita bahas langkah demi langkah:
""")

# LANGKAH 1: PERSIAPAN DATA
print("\n1️⃣ PERSIAPAN DATA & MODEL")
print("""
   Sebelum fine-tuning, kita melakukan:
   
   A. Persiapan Data:
      - Membaca file data (format: "teks####[triplet]")
      - Menghitung offset token untuk aspek dan opini
      - Membuat dataset dan dataloader
   
   B. Persiapan Model:
      - Memuat BERT pre-trained
      - Menambahkan komponen khusus untuk SPAN-ASTE:
        * Modul Span Representation
        * Span FFNN (klasifikasi span)
        * Target-Opinion Pair Representation
        * Pair FFNN (klasifikasi relasi)
   
   C. Konfigurasi Training:
      - Learning rate: 5e-5 untuk BERT, 1e-3 untuk komponen lain
      - Optimizer: AdamW
      - Jumlah epoch: ~10
      - Batch size: biasanya kecil (1-8) karena memori GPU
""")

# LANGKAH 2: SIKLUS TRAINING
print("\n2️⃣ SIKLUS TRAINING")
print("""
   Untuk setiap epoch (siklus pelatihan penuh):
   
   A. Training Phase:
      Untuk setiap batch data:
      
      1. Forward Pass:
         a. Encode teks dengan BERT
         b. Buat representasi span
         c. Klasifikasi span (aspek/opini)
         d. Pruning span
         e. Bentuk pasangan aspek-opini
         f. Klasifikasi relasi (sentimen)
      
      2. Perhitungan Loss:
         a. loss_ner: kesalahan dalam klasifikasi span
         b. loss_relation: kesalahan dalam klasifikasi relasi
         c. total_loss = 0.2 * loss_ner + loss_relation
      
      3. Backward Pass (Pembelajaran):
         a. Hitung gradien (seberapa harus mengubah model)
         b. Update parameter model
   
   B. Validation Phase:
      Saat validasi, kita:
      1. Jalankan model pada data validasi
      2. Hitung metrik evaluasi:
         - Precision: keakuratan prediksi
         - Recall: kelengkapan prediksi
         - F1-score: keseimbangan precision dan recall
      3. Menyimpan model jika F1-score meningkat
""")

# LANGKAH 3: VISUALISASI FORWARD PASS
print("\n3️⃣ DETAIL FORWARD PASS (DENGAN CONTOH)")
print("""
   Mari ilustrasikan forward pass dengan contoh sederhana:
   Teks: "aplikasi sangat bagus"
""")

# Simulasi representasi BERT
print("\n   A. Output BERT:")
bert_output = [
    [0.2, 0.3, -0.1],  # [CLS]
    [0.5, 0.2, 0.1],   # 'aplikasi'
    [0.3, 0.7, 0.2],   # 'sangat'
    [0.4, 0.6, 0.3],   # 'bagus'
    [0.1, 0.2, 0.1]    # [SEP]
]
print(f"      Token    | Representasi")
print(f"      ---------|-------------")
print(f"      [CLS]    | {bert_output[0]}")
print(f"      aplikasi | {bert_output[1]}")
print(f"      sangat   | {bert_output[2]}")
print(f"      bagus    | {bert_output[3]}")
print(f"      [SEP]    | {bert_output[4]}")

# Simulasi spans
print("\n   B. Beberapa Span yang Dibentuk:")
spans = [
    {"idx": (1, 1), "text": "aplikasi", "repr": [0.5, 0.2, 0.1, 0.5, 0.2, 0.1, 0.1]},
    {"idx": (2, 2), "text": "sangat", "repr": [0.3, 0.7, 0.2, 0.3, 0.7, 0.2, 0.1]},
    {"idx": (3, 3), "text": "bagus", "repr": [0.4, 0.6, 0.3, 0.4, 0.6, 0.3, 0.1]},
    {"idx": (2, 3), "text": "sangat bagus", "repr": [0.3, 0.7, 0.2, 0.4, 0.6, 0.3, 0.2]}
]

print(f"      Span         | Representasi")
print(f"      -------------|-------------")
for span in spans:
    print(f"      {span['text']:12} | [...{len(span['repr'])} nilai...]")

# Simulasi klasifikasi span
print("\n   C. Klasifikasi Span:")
span_scores = [
    {"text": "aplikasi", "scores": [0.1, 0.8, 0.1], "pred": "ASPECT"},
    {"text": "sangat", "scores": [0.7, 0.1, 0.2], "pred": "INVALID"},
    {"text": "bagus", "scores": [0.6, 0.1, 0.3], "pred": "INVALID"},
    {"text": "sangat bagus", "scores": [0.1, 0.2, 0.7], "pred": "OPINION"}
]

print(f"      Span         | INVALID | ASPECT | OPINION | Prediksi")
print(f"      -------------|---------|--------|---------|--------")
for score in span_scores:
    print(f"      {score['text']:12} | {score['scores'][0]:.1f}     | {score['scores'][1]:.1f}    | {score['scores'][2]:.1f}     | {score['pred']}")

# Simulasi pruning
print("\n   D. Pruning (Simpan Top Span):")
print(f"      Top ASPECT spans: ['aplikasi']")
print(f"      Top OPINION spans: ['sangat bagus']")

# Simulasi pasangan
print("\n   E. Pasangan dan Klasifikasi Relasi:")
pairs = [
    {"aspect": "aplikasi", "opinion": "sangat bagus", "scores": [0.05, 0.85, 0.05, 0.05], "pred": "POS"}
]

print(f"      Aspek    | Opini        | INVALID | POS   | NEG   | NEU   | Prediksi")
print(f"      ---------|--------------|---------|-------|-------|-------|--------")
for pair in pairs:
    print(f"      {pair['aspect']:8} | {pair['opinion']:12} | {pair['scores'][0]:.2f}     | {pair['scores'][1]:.2f}  | {pair['scores'][2]:.2f}  | {pair['scores'][3]:.2f}  | {pair['pred']}")

# LANGKAH 4: VISUALISASI PELATIHAN
print("\n4️⃣ VISUALISASI PELATIHAN & EVALUASI")

# Buat data simulasi
epochs = list(range(1, 11))
training_loss = [4.2, 3.5, 2.8, 2.3, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2]
validation_f1 = [0.35, 0.48, 0.55, 0.62, 0.67, 0.71, 0.72, 0.74, 0.75, 0.75]

# Plot simulasi training
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, 'b-o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot validation F1
plt.subplot(1, 2, 2)
plt.plot(epochs, validation_f1, 'g-o')
plt.title('Validation F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_visualization_tutorial.png')
print("\n   Grafik simulasi pelatihan disimpan sebagai 'training_visualization_tutorial.png'")
print("   - Kiri: Loss menurun selama pelatihan (model semakin baik)")
print("   - Kanan: F1-score meningkat (akurasi model semakin baik)")

# -----------------
# BAGIAN 6: PROSES VALIDASI SECARA DETAIL
# -----------------
print("\n\n MEMAHAMI PROSES VALIDASI")
print("=" * 70)

print("""
Validasi adalah cara untuk mengukur seberapa baik model pada data yang belum pernah dilihat.
Mari kita bahas proses validasi dengan contoh sederhana:
""")

print("\nCONTOH VALIDASI:")
valid_examples = [
    {
        "teks": "aplikasi sangat bagus",
        "gold": [("aplikasi", "sangat bagus", "POS")],
        "pred": [("aplikasi", "sangat bagus", "POS")]
    },
    {
        "teks": "gak bisa login",
        "gold": [("login", "gak bisa", "NEG")],
        "pred": [("login", "gak bisa", "NEG")]
    },
    {
        "teks": "server tidak terhubung",
        "gold": [("server", "tidak terhubung", "NEG")],
        "pred": []  # Tidak ada prediksi
    },
    {
        "teks": "tampilan bagus tapi lambat",
        "gold": [("tampilan", "bagus", "POS"), ("tampilan", "lambat", "NEG")],
        "pred": [("tampilan", "bagus", "POS")]  # Hanya 1 dari 2 yang terdeteksi
    }
]

print("\n1️⃣ PERHITUNGAN METRIK PER CONTOH:")
total_correct = 0
total_predicted = 0
total_gold = 0

for i, ex in enumerate(valid_examples):
    gold_set = set(ex["gold"])
    pred_set = set(ex["pred"])
    
    correct = len(gold_set.intersection(pred_set))
    total_correct += correct
    total_predicted += len(pred_set)
    total_gold += len(gold_set)
    
    print(f"\n   Contoh #{i+1}: \"{ex['teks']}\"")
    print(f"   Gold: {ex['gold']}")
    print(f"   Prediksi: {ex['pred']}")
    print(f"   Benar: {correct}, Prediksi: {len(pred_set)}, Gold: {len(gold_set)}")

# Hitung metrik
precision = total_correct / total_predicted if total_predicted > 0 else 0
recall = total_correct / total_gold if total_gold > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n2️⃣ METRIK KESELURUHAN:")
print(f"""
   Total Benar (True Positive): {total_correct}
   Total Prediksi: {total_predicted}
   Total Gold: {total_gold}
   
   Precision: {precision:.4f} = Benar / Total Prediksi
     - Seberapa akurat prediksi model
     - Dari semua triplet yang diprediksi, berapa persen yang benar?
   
   Recall: {recall:.4f} = Benar / Total Gold
     - Seberapa lengkap prediksi model
     - Dari semua triplet yang seharusnya ada, berapa persen yang berhasil diprediksi?
   
   F1 Score: {f1:.4f} = 2 * Precision * Recall / (Precision + Recall)
     - Keseimbangan antara precision dan recall
     - Semakin tinggi, semakin baik model
""")