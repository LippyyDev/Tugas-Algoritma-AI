# Tugas Algoritma Komputasi: Jaringan Saraf Tiruan dengan *Back Propagation*

Repositori ini berisi tugas mata kuliah Algoritma Komputasi, yaitu jaringan saraf tiruan *multilayer* yang dilatih dengan algoritma *back propagation* untuk mengenali 3 kelas objek dari citra *grayscale*. Algoritma ditulis manual per langkah dengan NumPy, mengikuti notasi variabel pada buku acuan.

## Ringkasan

| Aspek | Keterangan |
|-------|------------|
| Objek | Kendaraan |
| Kelas | Objek 1 = pesawat (*airplane*), Objek 2 = mobil (*automobile*), Objek 3 = kapal (*ship*) |
| Citra | 32 x 32 *pixel*, diubah ke *grayscale*, sehingga jumlah *node input* n = 1024 |
| *Hidden layer* | p *node* (jumlah masih ditentukan) |
| *Output* | Tepat 3 *node* dengan pola target `1 0 0`, `0 1 0`, dan `0 0 1` |
| Bahasa dan lingkungan | Python (NumPy dan Pillow) pada Google Colab |
| Buku acuan | Fausett, *Fundamentals of Neural Networks*, Bab 6, Langkah 0 sampai 9 dan *Application procedure* |

## Sumber *Dataset*

*Dataset* berasal dari **CIFAR-10**, *dataset* publik yang dibuat oleh Alex Krizhevsky, Vinod Nair, dan Geoffrey Hinton.

| Rincian | Keterangan |
|---------|------------|
| Halaman resmi | https://www.cs.toronto.edu/~kriz/cifar.html |
| Arsip yang dipakai | `cifar-10-binary.tar.gz` (versi biner), MD5 `c32a1d4ab5d03f1284b67883e8d87530` menurut halaman resmi |
| Alasan memakai versi biner | Memuat berkas *pickle* dari internet berisiko menjalankan kode asing |
| Kelas yang diambil | `airplane`, `automobile`, `ship` |
| Data latih | 300 gambar per kelas, diambil dari 5 *batch* latih CIFAR-10 |
| Data uji | 50 gambar per kelas, diambil dari *test set* resmi CIFAR-10 dan tidak ikut dilatih |
| Pengambilan acak | Memakai *seed* 42 agar dapat diulang |
| Konversi | Dari format biner menjadi berkas `.png` (RGB, 32 x 32) di Google Colab. Konversi ke *grayscale* dikerjakan pada tahap *preprocessing* di program *training* dan pengujian |

Sitasi:

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
- Fausett, L. (1994). *Fundamentals of Neural Networks: Architectures, Algorithms, and Applications*. Prentice Hall.

Hak atas gambar dan ketentuan penggunaan *dataset* mengikuti halaman resmi CIFAR-10 di atas.

## Isi *Dataset* di Repositori Ini

Sudah diperiksa pada berkas yang ada di repositori:

| Pemeriksaan | Hasil |
|-------------|-------|
| Total gambar | 1.050 berkas `.png` (900 latih dan 150 uji) |
| Ukuran gambar | Seluruhnya 32 x 32 *pixel* |
| Mode warna | Seluruhnya RGB |
| Duplikat berkas (berdasarkan SHA-256) | Tidak ada, termasuk antara data latih dan data uji |
| Ukuran total folder `DATASET` | Sekitar 4,2 MB |

## Struktur Folder

Nama folder dan berkas memakai huruf besar dengan garis bawah sebagai pengganti spasi. Ekstensi berkas tetap huruf kecil.

```
.
├── BOBOT/                        (hasil training: V, v_0, W, w_0 dalam 1 berkas)
├── DATASET/
│   ├── LATIH/
│   │   ├── KELAS1_PESAWAT/       (IMG001.png sampai IMG300.png)
│   │   ├── KELAS2_MOBIL/         (IMG001.png sampai IMG300.png)
│   │   └── KELAS3_KAPAL/         (IMG001.png sampai IMG300.png)
│   └── UJI/
│       ├── KELAS1_PESAWAT/       (IMG001.png sampai IMG050.png)
│       ├── KELAS2_MOBIL/         (IMG001.png sampai IMG050.png)
│       └── KELAS3_KAPAL/         (IMG001.png sampai IMG050.png)
├── DOKUMEN/
└── NOTEBOOK/                     (TRAINING.ipynb dan PENGUJIAN.ipynb)
```

## *Preprocessing*

Langkah yang sama dipakai pada program *training* dan program pengujian:

1. Ubah ke *grayscale*.
2. Ubah ukuran ke 32 x 32 *pixel* (hanya sebagai pengaman, karena gambar sudah seragam).
3. Ratakan (*flatten*) menjadi vektor berukuran n = 1024.
4. Normalisasi nilai *pixel* ke rentang 0 sampai 1 dengan membagi 255.

## Status Pengerjaan

| Berkas | Status |
|--------|--------|
| `DATASET/` | Sudah ada |
| `NOTEBOOK/TRAINING.ipynb` | Belum dibuat |
| `NOTEBOOK/PENGUJIAN.ipynb` | Belum dibuat |
| `BOBOT/` (berkas bobot) | Belum ada, dihasilkan setelah *training* |

## Catatan

- Resolusi CIFAR-10 sangat rendah (32 x 32), sehingga detail objek terbatas.
- Pesawat dan kapal sama-sama sering berlatar biru, sehingga kedua kelas ini mungkin sulit dibedakan oleh jaringan sederhana. Ini baru perkiraan dan belum diuji. Akurasi akan dilaporkan setelah *training* dan pengujian selesai.
- Pembaruan bobot dilakukan per pola latih sesuai buku, sehingga waktu *training* bertambah seiring jumlah gambar.
