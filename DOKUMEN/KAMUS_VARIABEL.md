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

Huruf besar (`X_i`, `Z_j`, `Y_k`) adalah *node*-nya. Huruf kecil (`x_i`, `z_j`, `y_k`) adalah nilai atau sinyal yang keluar dari *node* itu. Di tiap lapisan ada dua nilai: masukan bersih (`z_in_j`, `y_in_k`, hasil penjumlahan) dan keluaran (`z_j`, `y_k`, hasil sigmoid dari masukan bersih).

| Simbol buku | Nama di kode | Arti | Jumlah nilai |
|-------------|--------------|------|--------------|
| `X_i` | (indeks i) | *Node input* ke-i | 1024 *node* |
| `x_i` | `x` | Nilai *input* ke-i: nilai *pixel grayscale* dibagi 255, sehingga 0 sampai 1 | 1024 nilai |
| `Z_j` | (indeks j) | *Node hidden* ke-j | 100 *node* |
| `z_in_j` | `z_in` | Masukan bersih *node hidden* j (hasil jumlah, belum keluaran): `v_0j + SUM_i x_i v_ij` | 100 nilai |
| `z_j` | `z` | **Keluaran** *node hidden* j, dikirim ke *node output*: `f(z_in_j)`, selalu 0 sampai 1 | 100 nilai |
| `Y_k` | (indeks k) | *Node output* ke-k | 3 *node* |
| `y_in_k` | `y_in` | Masukan bersih *node output* k (hasil jumlah): `w_0k + SUM_j z_j w_jk` | 3 nilai |
| `y_k` | `y` | **Keluaran** *node output* k: `f(y_in_k)`, jawaban jaringan | 3 nilai |
| `t_k` | `t` | Target ke-k, kunci jawaban | 3 nilai |

Alur satu *node hidden*: `x` dijumlahkan menjadi `z_in`, lalu sigmoid mengubahnya menjadi `z`.

Pola target: pesawat `1 0 0`, mobil `0 1 0`, kapal `0 0 1`. Jadi `Y1` untuk pesawat, `Y2` untuk mobil, `Y3` untuk kapal. Kelas tebakan adalah *node output* dengan nilai terbesar.

## 4. Bobot dan *Bias* (Isi `BOBOT.npz`)

| Simbol buku | Nama di kode | Arti | Ukuran | Jumlah nilai | Kenapa segitu |
|-------------|--------------|------|--------|--------------|---------------|
| `v_ij` | `v` | Bobot dari *input* i ke *hidden* j | 1024 x 100 | 102.400 | Tiap dari 1024 *node input* terhubung ke tiap dari 100 *node hidden*. |
| `v_0j` | `v_0` | *Bias* (nilai tambahan tetap) *node hidden* j | 100 | 100 | Satu per *node hidden*. |
| `w_jk` | `w` | Bobot dari *hidden* j ke *output* k | 100 x 3 | 300 | Tiap dari 100 *node hidden* terhubung ke tiap dari 3 *node output*. |
| `w_0k` | `w_0` | *Bias* *node output* k | 3 | 3 | Satu per *node output*. |
| **Total** | | | | **102.803** | 102.400 + 100 + 300 + 3 |

Nilai awal (Step 0): `rng = np.random.default_rng(BIJI_ACAK)` membuat pembangkit bilangan acak, lalu `v`, `v_0`, `w`, `w_0` diisi `rng.uniform(-BATAS_ACAK, BATAS_ACAK, ukuran)`, yaitu acak antara -0,5 dan 0,5 (buku menyebut rentang ini sebagai pilihan umum). Ukurannya `(n, p)` untuk `v`, `p` untuk `v_0`, `(p, m)` untuk `w`, dan `m` untuk `w_0`.

## 5. Faktor Koreksi Galat (Step 6 dan 7)

Ada dua huruf Yunani yang mirip, tetapi artinya berbeda:

* **δ** (delta **kecil**, bentuk seperti bawang) adalah **faktor koreksi galat** milik satu *node*.
* **Δ** (delta **besar**, bentuk segitiga) adalah **besar perubahan bobot** yang akan ditambahkan pada bobot.

Hubungannya: `Δw_jk = α δ_k z_j`. Faktor koreksi galat δ dikalikan `α` dan `z_j` untuk menjadi besar perubahan bobot Δ.

| Simbol buku | Nama di kode | Arti | Jumlah nilai |
|-------------|--------------|------|--------------|
| `δ_k` (kecil) | `delta_k` | Faktor koreksi galat *node output* k: `(t_k - y_k) f'(y_in_k)` | 3 |
| `δ_in_j` (kecil) | `delta_in_j` | Jumlah `δ_k` yang dikirim mundur ke *node hidden* j: `SUM_k δ_k w_jk` | 100 |
| `δ_j` (kecil) | `delta_j` | Faktor koreksi galat *node hidden* j: `δ_in_j f'(z_in_j)` | 100 |
| `Δw_jk` (besar) | `delta_w` | Besar perubahan bobot W: `α δ_k z_j` | 300 |
| `Δw_0k` (besar) | `delta_w_0` | Besar perubahan *bias* W: `α δ_k` | 3 |
| `Δv_ij` (besar) | `delta_v` | Besar perubahan bobot V: `α δ_j x_i` | 102.400 |
| `Δv_0j` (besar) | `delta_v_0` | Besar perubahan *bias* V: `α δ_j` | 100 |

Di kode, δ dan Δ sama-sama dinamai `delta` karena nama variabel Python tidak bisa memakai simbol Yunani. Yang membedakan adalah akhiran namanya: `delta_k`, `delta_in_j`, `delta_j` adalah δ, sedangkan `delta_w`, `delta_w_0`, `delta_v`, `delta_v_0` adalah Δ.

Jumlah perubahan selalu sama dengan jumlah bobot yang diubah.

## 6. Parameter Pembelajaran

| Simbol di kode | Arti | Nilai | Catatan |
|----------------|------|-------|---------|
| `alpha` | *Learning rate*: besar langkah koreksi bobot | 0,1 | Nilai awal usulan, belum disetel. |
| `f(x)` | Fungsi aktivasi sigmoid biner: `1 / (1 + e^-x)` | | Mengubah nilai apa pun menjadi 0 sampai 1. |
| `f'(x)` | Turunan sigmoid: `f(x)(1 - f(x))` | | Dipakai menghitung koreksi galat. |
| `EPOCH_MAKS` | *Epoch* maksimum (satu *epoch* = satu putaran seluruh data latih) | 200 | Batas atas, belum disetel dari hasil *training* penuh. |
| `AMBANG_GALAT` | Berhenti bila galat rata-rata per pola di bawah angka ini | 0,01 | Belum disetel. |
| `BIJI_ACAK` | *Random seed* | 42 | Agar bobot awal dan urutan data bisa diulang. |
| `BATAS_ACAK` | Bobot awal diacak antara `-BATAS_ACAK` dan `+BATAS_ACAK` | 0,5 | Sesuai rentang yang disebut buku. |
| `rng` | Pembangkit bilangan acak, dibuat dari `BIJI_ACAK` | | Dipakai mengisi bobot awal dan mengacak urutan data tiap *epoch*. |

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
