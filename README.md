# Jaringan Saraf Tiruan *Back Propagation* untuk Mengenali Kendaraan

**Nama:** Muhammad Alif Qadri
**NIM:** D082261018
**Mata kuliah:** Algoritma Komputasi

Jaringan saraf tiruan *multilayer* yang dilatih dengan *back propagation* untuk mengenali tiga kelas kendaraan dari citra *grayscale*: pesawat, mobil, dan kapal. Algoritma ditulis manual per langkah dengan NumPy mengikuti Fausett, *Fundamentals of Neural Networks*, Bab 6 (Langkah 0 sampai 9 dan *Application procedure*), dengan notasi variabel yang sama dengan buku.

## Rancangan

| Aspek | Nilai |
|-------|-------|
| Kelas | pesawat (`1 0 0`), mobil (`0 1 0`), kapal (`0 0 1`) |
| *Node input* | n = 1024 (citra 32 x 32 *pixel*, *grayscale*) |
| *Node hidden* | p = 100 |
| *Node output* | m = 3 |
| Laju belajar | 0,1 |
| Berhenti | galat rata-rata di bawah 0,01 atau 200 *epoch* |
| Keputusan "dikenali" | nilai *output* terbesar minimal 0,5 |
| Pustaka | NumPy dan Pillow saja |

*Preprocessing* (sama di *training* dan pengujian): ubah ke *grayscale*, ubah ukuran ke 32 x 32 bila perlu, ratakan menjadi vektor 1024, bagi 255.

## Isi Repositori

| Berkas atau folder | Fungsi |
|--------------------|--------|
| `NOTEBOOK/TRAINING.ipynb` | Melatih jaringan di Google Colab dan menyimpan bobot ke satu berkas `BOBOT.npz` (`v`, `v_0`, `w`, `w_0`, `ukuran`) |
| `NOTEBOOK/PENGUJIAN.ipynb` | Memilih satu gambar, membaca `BOBOT.npz`, menghitung umpan maju saja, lalu menampilkan dikenali atau tidak |
| `DETEKSI/DETEKSI.html` | Halaman web satu berkas untuk mencoba model: pilih foto atau salah satu dari 10 contoh dataset uji. Bobot tertanam di halaman, jalan di peramban tanpa *server* |
| `VISUALISASI/VISUALISASI.html` | Penjelasan interaktif alur *training* dan pengujian dengan jaringan mini 16, 3, 3 (bukan model sebenarnya) |
| `DATASET/` | 900 gambar latih dan 150 gambar uji (`LATIH` dan `UJI`, per kelas) |
| `BOBOT/` | Tempat berkas bobot. Berkas `BOBOT.npz` hasil latihan disimpan di Google Drive dan belum ada di repositori ini |

## *Dataset*

Berasal dari **CIFAR-10** (Krizhevsky, 2009), versi biner dari https://www.cs.toronto.edu/~kriz/cifar.html (MD5 `c32a1d4ab5d03f1284b67883e8d87530` menurut halaman resmi). Kelas yang diambil: `airplane`, `automobile`, `ship`.

- Data latih: 300 gambar per kelas dari 5 *batch* latih.
- Data uji: 50 gambar per kelas dari *test set* resmi, tidak ikut dilatih.
- Pengambilan acak dengan *seed* 42. Gambar disimpan sebagai `.png` RGB 32 x 32.
- Diperiksa pada berkas di repositori: 1.050 gambar, semuanya 32 x 32 RGB, tidak ada duplikat (SHA-256), termasuk antara data latih dan uji.

Ketentuan penggunaan gambar mengikuti halaman resmi CIFAR-10.

## Hasil

Bobot dari Drive diuji pada 150 gambar uji (hitungan NumPy, dicocokkan dengan `DETEKSI.html`):

| Kelas | Tebakan benar | Dikenali benar (nilai ≥ 0,5) |
|-------|---------------|------------------------------|
| Pesawat | 33 dari 50 | 30 dari 50 |
| Mobil | 31 dari 50 | 24 dari 50 |
| Kapal | 27 dari 50 | 26 dari 50 |
| Total | 91 dari 150 (60,7%) | 80 dari 150 (53,3%) |

Artinya sekitar 4 dari 10 gambar uji salah atau tidak dikenali. Hal ini wajar untuk jaringan satu lapisan tersembunyi pada citra abu-abu beresolusi 32 x 32.

## Catatan

- Foto dengan bagian transparan atau resolusi jauh di atas 32 x 32 dapat memberi hasil yang kurang baik, karena model hanya dilatih pada citra CIFAR-10.
- Pembaruan bobot dilakukan per pola latih sesuai buku, sehingga waktu *training* bertambah seiring jumlah gambar.

## Sitasi

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
- Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications*. Prentice Hall.
