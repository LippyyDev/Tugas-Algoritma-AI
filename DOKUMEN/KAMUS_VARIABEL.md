# Kamus Variabel Jaringan Saraf Tiruan *Back Propagation*

Notasi mengikuti Fausett, *Fundamentals of Neural Networks* (1994), Bab 6. Angka dan nama kode sesuai `NOTEBOOK/TRAINING.ipynb` dan `NOTEBOOK/PENGUJIAN.ipynb`.

Arsitektur: **1024 *node input*  ->  100 *node hidden*  ->  3 *node output***.

Cara membaca: kolom **Simbol buku** adalah notasi di buku, kolom **Nama di kode** adalah nama variabel di *notebook*. Keduanya sengaja dibuat sama (`v_0j` di buku menjadi `v_0` di kode, `z_in_j` menjadi `z_in`, dan seterusnya). Huruf kapital `X`, `T` pada nama seperti `X_latih` berarti kumpulan banyak pola (matriks), bukan satu *node*.

---

## 1. Ukuran Jaringan

| Simbol | Arti | Jumlah | Dari mana angkanya |
|--------|------|--------|--------------------|
| `n` | Jumlah *node input* | **1024** | Gambar 32 x 32 *pixel*, setelah *grayscale* tiap *pixel* hanya satu nilai, sehingga 32 x 32 = 1024. Ketentuan dosen: jumlah *pixel* menentukan jumlah *node input*. Ukuran 32 x 32 adalah ukuran asli gambar CIFAR-10, tidak diubah. |
| `p` | Jumlah *node hidden* | **100** | Dosen membebaskan jumlahnya (contoh: 10 atau 100). Dipilih 100 oleh LIP. Belum diuji sebagai angka terbaik. |
| `m` | Jumlah *node output* | **3** | Ketentuan dosen: tepat 3, satu *node* untuk tiap kelas (pesawat, mobil, kapal). |
| `LEBAR`, `TINGGI` | Lebar dan tinggi gambar dalam *pixel* | 32 dan 32 | Ukuran asli CIFAR-10. `n = LEBAR x TINGGI`. |

## 2. Indeks (Penunjuk Nomor)

| Simbol | Menunjuk | Rentang | Jumlah |
|--------|----------|---------|--------|
| `i` | Nomor *node input* (nomor *pixel*) | 1 sampai n | 1024 |
| `j` | Nomor *node hidden* | 1 sampai p | 100 |
| `k` | Nomor *node output* (nomor kelas) | 1 sampai m | 3 |

Di buku indeks mulai dari 1. Di Python mulai dari 0, jadi `v_11` di buku adalah `v[0, 0]` di kode.

## 3. *Node* dan Sinyal

| Simbol buku | Nama di kode | Arti | Jumlah nilai |
|-------------|--------------|------|--------------|
| `X_i` | (tidak ada, hanya indeks `i`) | *Node input* ke-i | 1024 *node* |
| `x_i` | `x` | Nilai *input*: nilai *pixel grayscale* dibagi 255, sehingga 0 sampai 1 | 1024 nilai |
| `Z_j` | (tidak ada, hanya indeks `j`) | *Node hidden* ke-j | 100 *node* |
| `z_in_j` | `z_in` | Masukan bersih *node hidden*: `v_0j + SUM_i x_i v_ij` | 100 nilai |
| `z_j` | `z` | Keluaran *node hidden*: `f(z_in_j)` | 100 nilai |
| `Y_k` | (tidak ada, hanya indeks `k`) | *Node output* ke-k | 3 *node* |
| `y_in_k` | `y_in` | Masukan bersih *node output*: `w_0k + SUM_j z_j w_jk` | 3 nilai |
| `y_k` | `y` | Keluaran *node output*: `f(y_in_k)`, jawaban jaringan | 3 nilai |
| `t_k` | `t` | Target, kunci jawaban | 3 nilai |

Pola target: pesawat `1 0 0`, mobil `0 1 0`, kapal `0 0 1`. Jadi `Y1` untuk pesawat, `Y2` untuk mobil, `Y3` untuk kapal. Kelas tebakan adalah *node output* dengan nilai terbesar.

## 4. Bobot dan *Bias*

| Simbol buku | Nama di kode | Arti | Ukuran | Jumlah nilai | Kenapa segitu |
|-------------|--------------|------|--------|--------------|---------------|
| `v_ij` | `v` | Bobot dari *input* i ke *hidden* j | 1024 x 100 | 102.400 | Tiap dari 1024 *node input* terhubung ke tiap dari 100 *node hidden*. |
| `v_0j` | `v_0` | *Bias* (nilai tambahan tetap) *node hidden* j | 100 | 100 | Satu per *node hidden*. |
| `w_jk` | `w` | Bobot dari *hidden* j ke *output* k | 100 x 3 | 300 | Tiap dari 100 *node hidden* terhubung ke tiap dari 3 *node output*. |
| `w_0k` | `w_0` | *Bias* *node output* k | 3 | 3 | Satu per *node output*. |
| **Total** | | | | **102.803** | 102.400 + 100 + 300 + 3 |

Nilai awal: acak antara -0,5 dan 0,5 (buku menyebut rentang ini sebagai pilihan umum), dengan *random seed* 42 agar hasil bisa diulang. Pembangkit acaknya bernama `rng`.

## 5. Faktor Koreksi Galat (Step 6 dan 7)

| Simbol buku | Nama di kode | Arti | Ukuran | Jumlah nilai |
|-------------|--------------|------|--------|--------------|
| `delta_k` | `delta_k` | Faktor koreksi galat *node output*: `(t_k - y_k) f'(y_in_k)` | 3 | 3 |
| `delta_in_j` | `delta_in_j` | Jumlah `delta_k` yang dikirim mundur ke *node hidden*: `SUM_k delta_k w_jk` | 100 | 100 |
| `delta_j` | `delta_j` | Faktor koreksi galat *node hidden*: `delta_in_j f'(z_in_j)` | 100 | 100 |
| `Δw_jk` | `delta_w` | Selisih koreksi bobot W: `alpha delta_k z_j` | 100 x 3 | 300 |
| `Δw_0k` | `delta_w_0` | Selisih koreksi *bias* W: `alpha delta_k` | 3 | 3 |
| `Δv_ij` | `delta_v` | Selisih koreksi bobot V: `alpha delta_j x_i` | 1024 x 100 | 102.400 |
| `Δv_0j` | `delta_v_0` | Selisih koreksi *bias* V: `alpha delta_j` | 100 | 100 |

Jumlah koreksi selalu sama dengan jumlah bobot yang dikoreksi.

## 6. Parameter Pembelajaran

| Nama di kode | Arti | Nilai | Catatan |
|--------------|------|-------|---------|
| `alpha` | *Learning rate*: besar langkah koreksi bobot | 0,1 | Nilai awal usulan, belum disetel. |
| `EPOCH_MAKS` | *Epoch* maksimum (satu *epoch* = satu putaran seluruh data latih) | 200 | Batas atas, belum disetel dari hasil *training* penuh. |
| `AMBANG_GALAT` | Berhenti bila galat rata-rata per pola di bawah angka ini | 0,01 | Belum disetel. |
| `BATAS_ACAK` | Bobot awal diacak antara `-BATAS_ACAK` dan `+BATAS_ACAK` | 0,5 | Sesuai rentang yang disebut buku. |
| `BIJI_ACAK` | *Random seed* (angka awal pengacak) | 42 | Agar bobot awal dan urutan data bisa diulang. |
| `CETAK_TIAP` | Laporan dicetak tiap sekian *epoch* | 10 | Hanya mengatur tampilan, tidak memengaruhi hasil. |

Fungsi aktivasi: `f(x)` adalah sigmoid biner `1 / (1 + e^-x)`, mengubah nilai apa pun menjadi 0 sampai 1. `f_aksen(x)` adalah turunannya, `f(x)(1 - f(x))`.

## 7. Data

| Besaran | Jumlah | Kenapa segitu |
|---------|--------|---------------|
| Kelas | 3 | Ketentuan dosen: satu jenis objek, tiga jenis berbeda. Dipilih: pesawat, mobil, kapal. |
| Gambar latih | 900 (300 per kelas) | Dosen mewajibkan minimal 15 per kelas. Dipakai 300 agar jaringan punya cukup contoh. |
| Gambar uji | 150 (50 per kelas) | Diambil dari *test set* resmi CIFAR-10, tidak ikut dilatih, untuk mengukur kemampuan pada gambar baru. |
| Sumber | CIFAR-10 | Krizhevsky (2009). Ukuran seragam 32 x 32 *pixel*. |

Variabel penyimpan data di kode:

| Nama di kode | Arti | Ukuran |
|--------------|------|--------|
| `X_latih` | Seluruh gambar latih, tiap baris satu pola (1024 nilai `x`) | 900 x 1024 |
| `T_latih` | Target seluruh pola latih (`1 0 0`, `0 1 0`, atau `0 0 1`) | 900 x 3 |
| `label_latih` | Nomor kelas tiap pola latih (0, 1, 2), dipakai menghitung akurasi | 900 nilai |
| `nama_latih` | Nama berkas tiap gambar latih, contoh `KELAS1_PESAWAT/IMG001.png` | 900 nama |
| `X_uji`, `T_uji`, `label_uji` | Hal yang sama untuk data uji | 150 pola |
| `NAMA_KELAS` | Nama tiga folder kelas. Urutannya menentukan target (indeks 0 = `1 0 0`) | 3 nama |
| `jumlah_pola` | Jumlah pasangan latih | 900 |
| `s` | Nomor satu pola di data (indeks baris) | 0 sampai 899 |

Lokasi berkas: `FOLDER_PROYEK` (folder proyek di Google Drive), `FOLDER_LATIH` (folder gambar latih), `FOLDER_UJI` (folder gambar uji), `BERKAS_BOBOT` (alamat `BOBOT.npz`).

## 8. Fungsi di *Notebook*

| Fungsi | Isi singkat | Dipakai di |
|--------|-------------|------------|
| `praproses(gambar)` | Ubah satu gambar menjadi vektor `x`: *grayscale*, pengaman ukuran, *flatten*, dibagi 255 | *training* dan pengujian |
| `muat_dataset(folder)` | Baca semua gambar satu folder, hasilnya `X`, `T`, `label`, nama berkas | *training* |
| `f(x)`, `f_aksen(x)` | Sigmoid dan turunannya | *training* dan pengujian |
| `langkah_4`, `langkah_5`, `langkah_6`, `langkah_7`, `langkah_8` | Satu fungsi per step: Step 4, 5, 6, 7, 8 | *training* |
| `umpan_maju(X)` | Step 4 dan 5 untuk banyak pola, bobot tidak berubah | *training* (akurasi, evaluasi) |
| `akurasi(X, label)` | Persen tebakan benar | *training* |
| `laporan(X, label, judul)` | Cetak akurasi dan kembalikan tebakan | *training* |

## 9. Variabel Pelaporan dan Pembantu (*Training*)

| Nama di kode | Arti |
|--------------|------|
| `epoch` | Nomor putaran saat ini (1 sampai `EPOCH_MAKS`) |
| `galat_total` | Jumlah galat semua pola dalam satu *epoch*. Galat satu pola: `0,5 x SUM_k (t_k - y_k)^2` |
| `galat_rata` | `galat_total / jumlah_pola`, dibandingkan dengan `AMBANG_GALAT` di Step 9 |
| `riwayat_galat`, `riwayat_akurasi` | Catatan galat dan akurasi tiap *epoch*, untuk grafik |
| `keputusan`, `berhenti` | Hasil uji syarat berhenti (Step 9) |
| `waktu_mulai` | Waktu mulai *training*, untuk menghitung lama |
| `hasil` | Matriks keluaran `y` semua pola di dalam `umpan_maju` |
| `tebakan`, `tebakan_uji` | Nomor kelas yang ditebak jaringan (*node output* terbesar) |
| `benar` | Persen tebakan benar di dalam `laporan` |
| `y_uji` | Keluaran jaringan untuk seluruh data uji |
| `v_s`, `v_0_s`, `w_s`, `w_0_s` | Salinan bobot untuk contoh Step 8 di Sel 10, agar bobot asli tidak berubah |
| `lama_`, `baru_` | Nilai bobot sebelum dan sesudah Step 8 pada contoh Sel 10 |
| `cek`, `kunci`, `sama` | Membaca ulang `BOBOT.npz` dan memeriksa isinya sama dengan bobot di memori |
| `judul`, `baris`, `persen`, `t_teks`, `y_teks`, `pola` | Teks dan angka bantu untuk mencetak tabel |
| `awal` | Nomor *pixel* pertama pada tiap baris saat mencetak seluruh `x` di Sel 5 |
| `fig`, `sumbu`, `a1`, `a2`, `a3` | Objek grafik (*matplotlib*) |
| `asli`, `abu`, `gambar` | Gambar berwarna asli, gambar *grayscale*, dan gambar yang sedang diproses |
| `folder`, `daftar`, `berkas`, `nama`, `nama_berkas`, `target`, `larik`, `a`, `b`, `_` | Penampung sementara di dalam *loop* atau fungsi |

## 10. Isi Berkas `BOBOT.npz`

Hasil *training*, dibaca oleh program pengujian.

| Kunci | Isi | Ukuran |
|-------|-----|--------|
| `v` | Bobot V (*input* ke *hidden*) | 1024 x 100 |
| `v_0` | *Bias* *node hidden* | 100 |
| `w` | Bobot W (*hidden* ke *output*) | 100 x 3 |
| `w_0` | *Bias* *node output* | 3 |
| `ukuran` | Lebar dan tinggi gambar saat *training*, agar pengujian memakai ukuran yang sama | 2 nilai (32 dan 32) |

## 11. Variabel Khusus Program Pengujian

| Nama di kode | Arti | Nilai |
|--------------|------|-------|
| `AMBANG_DIKENALI` | *Threshold*: nilai *output* terbesar minimal sekian agar gambar dinyatakan dikenali | 0,5 (belum dikalibrasi) |
| `NAMA_KELAS` | Nama kelas untuk `Y1`, `Y2`, `Y3` | `PESAWAT`, `MOBIL`, `KAPAL` |
| `KELAS_SEBENARNYA` | Opsional: kelas gambar yang diunggah, bila diketahui (0, 1, atau 2) | `None` bila tidak diisi |
| `bobot`, `sumber` | Isi `BOBOT.npz` yang dibaca, dan asalnya (Drive atau unggahan) | |
| `unggahan`, `nama_berkas` | Hasil `files.upload()` dan nama berkas gambar yang dipilih | |
| `gambar` | Gambar yang diunggah | |
| `x`, `y` | Vektor *input* gambar dan keluaran jaringan (`Y1`, `Y2`, `Y3`) | 1024 nilai dan 3 nilai |
| `indeks` | Nomor *node output* dengan nilai terbesar (kelas tebakan) | 0, 1, atau 2 |
| `nilai_maks` | Nilai `y` terbesar | |
| `dikenali` | `True` bila `nilai_maks` mencapai `AMBANG_DIKENALI` | |
| `cocok` | `True` bila tebakan sama dengan `KELAS_SEBENARNYA` dan dikenali | |
| `tanda`, `warna`, `fig`, `a1`, `a2`, `a3` | Penanda teks dan warna, serta objek grafik | |

`v`, `v_0`, `w`, `w_0`, `n`, `p`, `m`, `LEBAR`, `TINGGI` dibaca dari `BOBOT.npz`, bukan ditetapkan manual. `n`, `p`, `m` dihitung dari ukuran bobot.

## 12. Letak Tiap Step di `TRAINING.ipynb`

| Step | Isi singkat | Sel |
|------|-------------|-----|
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
