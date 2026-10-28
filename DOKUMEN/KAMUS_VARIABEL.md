# Kamus Variabel *Back Propagation*

Notasi mengikuti Fausett (1994), Bab 6. Nama di kode sengaja sama dengan notasi buku.
Arsitektur: **1024 *node input*  ->  100 *node hidden*  ->  3 *node output***.

## 1. Variabel Utama

| Buku | Kode | Arti | Jumlah | Asal angka |
|------|------|------|--------|------------|
| `n` | `n` | *Node input* (satu per *pixel*) | **1024** | 32 x 32 *pixel*, ukuran asli gambar CIFAR-10. Aturan dosen: jumlah *pixel* = jumlah *node input*. |
| `p` | `p` | *Node hidden* | **100** | Dosen membebaskan (contoh 10 atau 100). Dipilih 100 oleh LIP, belum diuji sebagai yang terbaik. |
| `m` | `m` | *Node output* (satu per kelas) | **3** | Aturan dosen: tepat 3. |
| `i`, `j`, `k` | `i`, `j`, `k` | Nomor *node input*, *hidden*, *output* | 1 sampai 1024, 100, 3 | Di buku mulai dari 1, di Python dari 0 (`v_11` = `v[0, 0]`). |
| `x_i` | `x` | Nilai *pixel grayscale* dibagi 255 (0 sampai 1) | 1024 | Satu nilai per *pixel*. |
| `z_in_j` | `z_in` | Masukan bersih *hidden*: `v_0j + SUM_i x_i v_ij` | 100 | Satu per *node hidden*. |
| `z_j` | `z` | Keluaran *hidden*: `f(z_in_j)` | 100 | Satu per *node hidden*. |
| `y_in_k` | `y_in` | Masukan bersih *output*: `w_0k + SUM_j z_j w_jk` | 3 | Satu per *node output*. |
| `y_k` | `y` | Keluaran *output*, jawaban jaringan: `f(y_in_k)` | 3 | Satu per kelas. |
| `t_k` | `t` | Target (kunci jawaban) | 3 | Pesawat `1 0 0`, mobil `0 1 0`, kapal `0 0 1`. |
| `v_ij` | `v` | Bobot *input* ke *hidden* | 1024 x 100 = **102.400** | Tiap *node input* terhubung ke tiap *node hidden*. |
| `v_0j` | `v_0` | *Bias* (nilai tambahan tetap) *hidden* | **100** | Satu per *node hidden*. |
| `w_jk` | `w` | Bobot *hidden* ke *output* | 100 x 3 = **300** | Tiap *node hidden* terhubung ke tiap *node output*. |
| `w_0k` | `w_0` | *Bias* *output* | **3** | Satu per *node output*. |
| `delta_k` | `delta_k` | Faktor koreksi galat *output*: `(t_k - y_k) f'(y_in_k)` | 3 | Satu per *node output*. |
| `delta_j` | `delta_j` | Faktor koreksi galat *hidden*: `delta_in_j f'(z_in_j)` | 100 | Satu per *node hidden*. |
| `Δw_jk`, `Δw_0k` | `delta_w`, `delta_w_0` | Koreksi bobot dan *bias* W | 300 dan 3 | Ukuran sama dengan `w` dan `w_0`. |
| `Δv_ij`, `Δv_0j` | `delta_v`, `delta_v_0` | Koreksi bobot dan *bias* V | 102.400 dan 100 | Ukuran sama dengan `v` dan `v_0`. |

Total bobot dan *bias*: 102.400 + 100 + 300 + 3 = **102.803** nilai. Hasil *training* ini disimpan di `BOBOT.npz` dengan kunci `v`, `v_0`, `w`, `w_0`, dan `ukuran` (lebar dan tinggi gambar, 32 dan 32).

## 2. Parameter

| Kode | Arti | Nilai | Catatan |
|------|------|-------|---------|
| `alpha` | *Learning rate*: besar langkah koreksi bobot | 0,1 | Usulan awal, belum disetel. |
| `f`, `f_aksen` | Fungsi sigmoid `1 / (1 + e^-x)` dan turunannya `f(x)(1 - f(x))` | | Mengubah nilai apa pun menjadi 0 sampai 1. |
| `EPOCH_MAKS` | *Epoch* maksimum (satu *epoch* = satu putaran seluruh data latih) | 200 | Batas atas, belum disetel. |
| `AMBANG_GALAT` | Berhenti bila galat rata-rata di bawah angka ini | 0,01 | Belum disetel. |
| `BATAS_ACAK` | Bobot awal diacak antara -0,5 dan 0,5 | 0,5 | Rentang yang disebut buku. |
| `BIJI_ACAK` | *Random seed* (angka awal pengacak, agar hasil bisa diulang) | 42 | `rng` adalah pembangkit acak yang memakainya. |

## 3. Data

* 3 kelas (pesawat, mobil, kapal), ketentuan dosen: satu jenis objek, tiga jenis berbeda.
* 900 gambar latih (300 per kelas): dosen mewajibkan minimal 15 per kelas.
* 150 gambar uji (50 per kelas): dari *test set* resmi CIFAR-10, tidak ikut dilatih.
* Di kode: `X_latih` (900 x 1024), `T_latih` (900 x 3), `X_uji`, `T_uji` untuk data uji.

## 4. Letak Tiap Step di `TRAINING.ipynb`

| Step | Isi singkat | Sel |
|------|-------------|-----|
| 0 | Isi bobot dengan bilangan acak kecil | 4 |
| 1, 2, 9 | Ulangi per *epoch*, tiap pasangan latih, uji syarat berhenti | 11 |
| 3 | *Input unit* menerima `x_i` | 5 |
| 4 | Hitung `z_in_j`, `z_j` | 6 |
| 5 | Hitung `y_in_k`, `y_k` | 7 |
| 6 | Hitung `delta_k`, `Δw_jk`, `Δw_0k` | 8 |
| 7 | Hitung `delta_j`, `Δv_ij`, `Δv_0j` | 9 |
| 8 | Perbarui bobot: baru = lama + koreksi | 10 |

Step 3 sampai 5 adalah umpan maju (*feedforward*), Step 6 sampai 8 adalah propagasi galat mundur. Program pengujian hanya memakai umpan maju.
