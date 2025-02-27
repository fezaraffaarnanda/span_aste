# Progress Skripsi (Februari) - Aspect Sentiment Triplet Extraction (ASTE)

## Hal yang Sudah Dilakukan

### 1. Pembuatan Dataset

Dataset sudah selesai dianotasi oleh total 5 annotator yang dibagi menjadi 2 tim. Untuk memastikan kualitas anotasi, telah dihitung *Inter-Annotator Agreement* menggunakan Cohen's Kappa dengan hasil sebagai berikut:

| Elemen Anotasi | Rata-rata Cohen's Kappa |
|----------------|-------------------------|
| Aspect         | 0.85                    |
| Opinion        | 0.76                    |
| Sentiment      | 0.84                    |

Nilai Cohen's Kappa di atas 0.75 menunjukkan kesepakatan antar annotator yang sangat baik, yang menandakan dataset yang dihasilkan memiliki kualitas dan konsistensi yang tinggi.

### 2. Fine-Tuning Model

Fine-tuning model Span-ASTE sudah dilakukan dan sedang dalam proses evaluasi menggunakan metode *5-fold cross validation*. Hasil sementara untuk fold 1, 2, dan 3 menunjukkan rata-rata F1-score sebesar 0.61.

Berdasarkan literatur dan penelitian sebelumnya, nilai ini termasuk dalam rentang yang wajar dan dapat diterima karena:

- Sebagian besar implementasi ASTE memiliki nilai F1-score berkisar antara 0.5 – 0.6 pada berbagai arsitektur
- Untuk arsitektur Span-ASTE khususnya, nilai F1-score sebesar 0.61 sudah cukup baik dibandingkan dengan benchmark pada dataset serupa seperti res14 (domain restaurant) yang mencapai F1-score 0.59

## Hal yang Akan Dilakukan

### 1. Arsitektur Model Span-ASTE

Mempelajari arsitektur model Span-ASTE secara mendalam, meliputi:
- Proses transformasi input
- Mekanisme kerja di setiap komponen model
- Metode perhitungan loss
- Metrik evaluasi yang digunakan
- Proses pembentukan output triplet hasil ekstraksi

### 2. Perbaikan Label

Melakukan *cleaning* pada hasil konversi label dari format LabelStudio menjadi format Span-ASTE, karena saat ini masih terdapat beberapa teks yang belum sesuai dengan hasil konversinya. Perbaikan ini penting untuk memastikan model dilatih dengan data yang akurat.

### 3. Dokumentasi dan Evaluasi

- Menyimpan rekapitulasi lengkap hasil 5-fold cross validation untuk setiap langkah proses
- Memodifikasi sistem validasi dengan mengubah parameter validation steps menjadi per-epoch untuk mendapatkan evaluasi yang lebih stabil dan konsisten
