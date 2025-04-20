

## Laporan Progress Skripsi - Maret

### Pembaruan Progress

#### Update Hasil Labelling:
- Memisahkan tanda baca menjadi token tersendiri.
- Contoh:
  - **Sebelum:** "Tampilannya bagus, namun server lemot."
    - Label: `[(\0, \1, POS), (\3, \4, NEG)]`
  - **Sesudah:** "Tampilannya bagus , namun server lemot."
    - Label: `[(\0, \1, POS), (\4, \5, NEG)]`
- Perubahan ini sesuai dengan indeks label pada dataset SemEval 2016 untuk fine-tuning model Span-ASTE.

#### Pembersihan Label:
- Membersihkan tanda baca berlebihan yang dapat memengaruhi indeks token.
- Contoh: tanda titik lebih dari 2 akan dipotong menjadi satu saja.

### Hasil Fine-tuning

| Model                 | F1 Score | Waktu (menit) |
|-----------------------|----------|---------------|
| IndoBERT-base        | 68.06    | 1160          |
| IndoBERT-large       | 74.06    | 7281          |
| IndoBERT-lite-base   | 70.86    | 9149          |
| IndoBERT-lite-large   | 68.70    | 254          |

- **IndoBERT-large** memiliki akurasi tertinggi tetapi dengan beban komputasi tinggi (~5 jam).
- **IndoBERT-lite-base** menunjukkan performa lebih baik dibanding IndoBERT-base dengan parameter 10x lebih sedikit.
- Performa **F1 score setara** dengan paper original Span-ASTE (rata-rata **67.69**).
- **Update label dengan memperhatikan tanda baca meningkatkan F1 score** dari **59-61 menjadi 68** (menggunakan IndoBERT-base).

### Analisis Model IndoBERT-lite-base

- Menggunakan arsitektur **ALBERT (A-Lite BERT)** yang mengurangi jumlah parameter.
- Dilatih menggunakan **SOP (Sentence Order Prediction)** yang lebih unggul dalam menangkap urutan kata dibandingkan **NSP (Next Sentence Prediction)** pada BERT.
- Sesuai dengan arsitektur **Span-ASTE** yang menghitung **width embedding** dan **distance embedding** untuk menentukan valid pair relation.

### Proses Kerja Span-Level ASTE

1. **Sentence Encoding**: Setiap input diencoding menggunakan **BERT tokenizer** (subword/wordpiece).
2. **Span Representation**: Membentuk representasi span dari kemungkinan kata dalam kalimat.
   - **Contoh:** "server sangat cepat"
   - **Kemungkinan span:**
     - "server", "sangat", "cepat"
     - "server sangat", "sangat cepat"
     - "server sangat cepat"
3. **Aspect Term Extraction dan Opinion Term Extraction**:
   - Mengklasifikasikan setiap token ke dalam **"TARGET", "OPINION", atau "INVALID"**.
4. **Pruning Target dan Opinion**:
   - Mengambil **top n** untuk efisiensi komputasi.
5. **Triplet Module**:
   - Membentuk pasangan valid dengan label **"POSITIVE", "NEGATIVE", "NEUTRAL", dan "INVALID"**.
6. **Loss Function**:
   - Menggunakan **negative log likelihood** dengan rumus:
     ```
     Loss = 0.2 × loss_NER + loss_RELATION
     ```

### Catatan Penelitian

- **Penelitian ini merupakan yang pertama** membangun dataset aspect sentiment triplet extraction berbahasa Indonesia.
- **Penelitian ini juga pertama kali membangun model Span-ASTE untuk Bahasa Indonesia**.
- **Domain yang digunakan** adalah **aplikasi pemerintahan berbahasa Indonesia**.

---