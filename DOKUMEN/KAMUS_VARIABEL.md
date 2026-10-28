# Kamus Variabel Jaringan Saraf Tiruan *Back Propagation*

Notasi mengikuti Fausett, *Fundamentals of Neural Networks* (1994), Bab 6. Angka sesuai `NOTEBOOK/TRAINING.ipynb`.

Arsitektur: **1024 *node input*  ->  100 *node hidden*  ->  3 *node output***.

---

## 1. Ukuran Jaringan

| Simbol | Arti | Jumlah | Dari mana angkanya |
|--------|------|--------|--------------------|
| `n` | Jumlah *node input* | **1024** | Gambar 32 x 32 *pixel*, setelah *grayscale* tiap *pixel* hanya satu nilai, sehingga 32 x 32 = 1024. Ketentuan dosen: jumlah *pixel* menentukan jumlah *node input*. Ukuran 32 x 32 adalah ukuran asli gambar CIFAR-10, tidak diubah. |
| `p` | Jumlah *node hidden* | **100** | Dosen membebaskan jumlahnya (contoh: 10 atau 100). Dipilih 100 oleh LIP. Belum diuji sebagai angka terbaik. |
| `m` | Jumlah *node output* | **3** | Ketentuan dosen: tepat 3, satu *node* untuk tiap kelas (pesawat, mobil, kapal). |

## 2. Indeks (Penunjuk Nomor)

| Simbol | Menunjuk | Rentang | Jumlah |
|--------|----------|---------|--------|
| `i` | Nomor *node input* (nomor *pixel*) | 1 sampai n | 1024 |
| `j` | Nomor *node hidden* | 1 sampai p | 100 |
| `k` | Nomor *node output* (nomor kelas) | 1 sampai m | 3 |

Di buku indeks mulai dari 1. Di Python mulai dari 0, jadi `v_11` di buku adalah `v[0, 0]` di kode.

## 3. *Node* dan Sinyal

| Simbol | Arti | Jumlah nilai |
|--------|------|--------------|
| `X_i` | *Node input* ke-i | 1024 *node* |
| `x_i` | Nilai *input* ke-i: nilai *pixel grayscale* dibagi 255, sehingga 0 sampai 1 | 1024 nilai |
| `Z_j` | *Node hidden* ke-j | 100 *node* |
| `z_in_j` | Masukan bersih *node hidden* j: `v_0j + SUM_i x_i v_ij` | 100 nilai |
| `z_j` | Keluaran *node hidden* j: `f(z_in_j)` | 100 nilai |
| `Y_k` | *Node output* ke-k | 3 *node* |
| `y_in_k` | Masukan bersih *node output* k: `w_0k + SUM_j z_j w_jk` | 3 nilai |
| `y_k` | Keluaran *node output* k: `f(y_in_k)`, jawaban jaringan | 3 nilai |
| `t_k` | Target ke-k, kunci jawaban | 3 nilai |

Pola target: pesawat `1 0 0`, mobil `0 1 0`, kapal `0 0 1`. Jadi `Y1` untuk pesawat, `Y2` untuk mobil, `Y3` untuk kapal. Kelas tebakan adalah *node output* dengan nilai terbesar.

## 4. Bobot dan *Bias* (Isi `BOBOT.npz`)

| Simbol | Arti | Ukuran | Jumlah nilai | Kenapa segitu |
|--------|------|--------|--------------|---------------|
| `v_ij` | Bobot dari *input* i ke *hidden* j | 1024 x 100 | 102.400 | Tiap dari 1024 *node input* terhubung ke tiap dari 100 *node hidden*. |
| `v_0j` | *Bias* (nilai tambahan tetap) *node hidden* j | 100 | 100 | Satu per *node hidden*. |
| `w_jk` | Bobot dari *hidden* j ke *output* k | 100 x 3 | 300 | Tiap dari 100 *node hidden* terhubung ke tiap dari 3 *node output*. |
| `w_0k` | *Bias* *node output* k | 3 | 3 | Satu per *node output*. |
| **Total** | | | **102.803** | 102.400 + 100 + 300 + 3 |

Nilai awal: acak antara -0,5 dan 0,5 (buku menyebut rentang ini sebagai pilihan umum), dengan *random seed* 42 agar hasil bisa diulang.

## 5. Faktor Koreksi Galat (Step 6 dan 7)

| Simbol | Arti | Jumlah nilai |
|--------|------|--------------|
| `delta_k` | Faktor koreksi galat *node output* k: `(t_k - y_k) f'(y_in_k)` | 3 |
| `delta_in_j` | Jumlah `delta_k` yang dikirim mundur ke *node hidden* j: `SUM_k delta_k w_jk` | 100 |
| `delta_j` | Faktor koreksi galat *node hidden* j: `delta_in_j f'(z_in_j)` | 100 |
| `Δw_jk` | Selisih koreksi bobot W: `alpha delta_k z_j` | 300 |
| `Δw_0k` | Selisih koreksi *bias* W: `alpha delta_k` | 3 |
| `Δv_ij` | Selisih koreksi bobot V: `alpha delta_j x_i` | 102.400 |
| `Δv_0j` | Selisih koreksi *bias* V: `alpha delta_j` | 100 |

Jumlah koreksi selalu sama dengan jumlah bobot yang dikoreksi.

## 6. Parameter Pembelajaran

| Simbol di kode | Arti | Nilai | Catatan |
|----------------|------|-------|---------|
| `alpha` | *Learning rate*: besar langkah koreksi bobot | 0,1 | Nilai awal usulan, belum disetel. |
| `f(x)` | Fungsi aktivasi sigmoid biner: `1 / (1 + e^-x)` | | Mengubah nilai apa pun menjadi 0 sampai 1. |
| `f'(x)` | Turunan sigmoid: `f(x)(1 - f(x))` | | Dipakai menghitung koreksi galat. |
| `EPOCH_MAKS` | *Epoch* maksimum (satu *epoch* = satu putaran seluruh data latih) | 200 | Batas atas, belum disetel dari hasil *training* penuh. |
| `AMBANG_GALAT` | Berhenti bila galat rata-rata per pola di bawah angka ini | 0,01 | Belum disetel. |
| `BIJI_ACAK` | *Random seed* | 42 | Agar bobot awal dan urutan data bisa diulang. |

Galat satu pola: `0,5 x jumlah (t_k - y_k)^2`. Galat rata-rata adalah rata-ratanya atas 900 pola.

## 7. Data

| Besaran | Jumlah | Kenapa segitu |
|---------|--------|---------------|
| Kelas | 3 | Ketentuan dosen: satu jenis objek, tiga jenis berbeda. Dipilih: pesawat, mobil, kapal. |
| Gambar latih | 900 (300 per kelas) | Dosen mewajibkan minimal 15 per kelas. Dipakai 300 agar jaringan punya cukup contoh. |
| Gambar uji | 150 (50 per kelas) | Diambil dari *test set* resmi CIFAR-10, tidak ikut dilatih, untuk mengukur kemampuan pada gambar baru. |
| Sumber | CIFAR-10 | Krizhevsky (2009). Ukuran seragam 32 x 32 *pixel*. |

## 8. Letak Tiap Step di *Notebook*

| Step | Isi singkat | Sel di `TRAINING.ipynb` |
|------|-------------|-------------------------|
| 0 | Isi bobot dengan bilangan acak kecil | Sel 4 |
| 1 | Ulangi selama syarat berhenti belum terpenuhi | Sel 11 |
| 2 | Untuk tiap pasangan latih (x, t), kerjakan Step 3 sampai 8 | Sel 11 |
| 3 | *Input unit* menerima `x_i` | Sel 5 |
| 4 | Hitung `z_in_j` dan `z_j` (*hidden layer*) | Sel 6 |
| 5 | Hitung `y_in_k` dan `y_k` (*output layer*) | Sel 7 |
| 6 | Hitung `delta_k`, `Δw_jk`, `Δw_0k` | Sel 8 |
| 7 | Hitung `delta_in_j`, `delta_j`, `Δv_ij`, `Δv_0j` | Sel 9 |
| 8 | Perbarui bobot: baru = lama + koreksi | Sel 10 |
| 9 | Uji syarat berhenti | Sel 11 |

Step 3 sampai 5 adalah umpan maju (*feedforward*). Step 6 sampai 8 adalah propagasi galat mundur (*backpropagation of error*). Program pengujian hanya memakai umpan maju.
