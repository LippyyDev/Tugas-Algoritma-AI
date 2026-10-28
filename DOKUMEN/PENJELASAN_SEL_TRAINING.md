# Penjelasan Tiap Sel `TRAINING.ipynb`

Dokumen ini menjelaskan isi dan cara kerja tiap sel. Sel 1 sampai 4 dan Sel 12 sampai 15 dijelaskan singkat. Sel 5 sampai 11 (Step 1 sampai 9, inti *back propagation*) dijelaskan rinci.

Angka contoh berasal dari satu kali menjalankan *notebook* dengan *random seed* 42, gambar contoh `KELAS1_PESAWAT/IMG001.png`, dan bobot awal. Bila Anda tidak mengubah apa pun, angka di Colab seharusnya sama.

---

## Gambaran Besar

1. Sel 1 sampai 3: menyiapkan pengaturan, data, dan memeriksa gambar.
2. Sel 4: **Step 0**, mengisi bobot awal secara acak.
3. Sel 5 sampai 10: memperlihatkan **Step 3 sampai 8** satu per satu pada **satu gambar contoh**. Pada tahap ini jaringan belum belajar.
4. Sel 11: **Step 1, 2, dan 9**. Di sinilah *training* sesungguhnya, memanggil fungsi Step 4 sampai 8 untuk **900 gambar**, berulang tiap *epoch*.
5. Sel 12 sampai 15: grafik, evaluasi, pemeriksaan keluaran, dan penyimpanan bobot.

Satu gambar melewati urutan ini:

```
gambar (x, 1024 nilai)
   Step 3:  x diterima node input
   Step 4:  x  -> node hidden  -> z_in, z      (100 nilai)
   Step 5:  z  -> node output  -> y_in, y      (3 nilai)
   Step 6:  bandingkan y dengan target t -> delta_k, koreksi untuk w
   Step 7:  kirim delta_k mundur           -> delta_j, koreksi untuk v
   Step 8:  bobot baru = bobot lama + koreksi
```

Step 3 sampai 5 disebut umpan maju (*feedforward*), Step 6 sampai 8 disebut propagasi galat mundur (*backpropagation of error*).

---

## Sel 1. Pengaturan Awal dan Konfigurasi Jaringan

* Menyambungkan Colab ke Google Drive (`drive.mount`) dan menetapkan alamat folder data latih, data uji, dan berkas `BOBOT.npz`.
* Menetapkan ukuran jaringan: `n = 1024`, `p = 100`, `m = 3`.
* Menetapkan parameter pembelajaran: `alpha = 0,1`, `EPOCH_MAKS = 200`, `AMBANG_GALAT = 0,01`, `BATAS_ACAK = 0,5`, `BIJI_ACAK = 42`, `CETAK_TIAP = 10`.
* `NAMA_KELAS` menentukan target: urutan pertama (`KELAS1_PESAWAT`) berarti `1 0 0`, kedua `0 1 0`, ketiga `0 0 1`.
* *Output*: tabel pengaturan.

## Sel 2. Prapemrosesan Citra dan Pemuatan *Dataset*

* `praproses(gambar)` mengubah satu gambar menjadi vektor `x` dalam 4 langkah: *grayscale*, pengaman ukuran 32 x 32, *flatten* (gambar 2 dimensi menjadi satu baris 1024 nilai), lalu dibagi 255 agar bernilai 0 sampai 1.
* `muat_dataset(folder)` membaca semua gambar per kelas (berkas diurutkan dengan `sorted`), memanggil `praproses`, dan membuat target dari urutan kelas (*one-hot*: hanya satu nilai 1).
* Hasilnya `X_latih` (900 x 1024), `T_latih` (900 x 3), `label_latih`, dan `nama_latih`, serta pasangannya untuk data uji.
* *Output*: jumlah gambar tiap kelas (300 latih, 50 uji) dan target tiap kelas.

## Sel 3. Verifikasi Konversi Citra

* Menampilkan gambar pertama tiap kelas: baris atas berwarna asli (mode RGB), baris bawah hasil `praproses` (*grayscale*).
* Tujuannya hanya memastikan konversi benar: bentuk gambar sama, warnanya hilang.

## Sel 4. Fungsi Aktivasi dan Step 0

* `f(x) = 1 / (1 + e^-x)`: sigmoid, mengubah nilai apa pun menjadi 0 sampai 1. `f_aksen(x) = f(x)(1 - f(x))`: turunannya, dipakai di Step 6 dan 7.
* **Step 0**: `rng` (pembangkit acak dari seed 42) mengisi `v`, `v_0`, `w`, `w_0` dengan bilangan acak antara -0,5 dan 0,5.
* *Output*: tabel ukuran bobot (1024 x 100, 100, 100 x 3, 3) dan rentang bobot awal.

---

## Sel 5. Step 3: Pola Contoh dan Sinyal *Input* (rinci)

**Rumus buku:** *Each input unit (X_i, i = 1, ..., n) receives input signal x_i and broadcasts this signal to all units in the layer above (the hidden units).*

**Yang dikerjakan sel ini**
* `x = X_latih[0]` mengambil **satu** gambar latih (gambar pertama, pesawat) sebagai **pola contoh**, dan `t = T_latih[0]` mengambil targetnya (`1 0 0`).
* Sel 5 sampai 10 memakai gambar yang sama supaya tiap step bisa diikuti satu per satu. Pada tahap ini bobot belum belajar apa pun.
* Step 3 sendiri tidak punya hitungan: tiap *node input* hanya menerima satu nilai `x_i` lalu meneruskannya ke semua *node hidden*.

**Cara membaca *output***
* Gambar 32 x 32 yang sedang dibaca tampil lebih dulu.
* `x_i` adalah kecerahan *pixel* ke-i: 0 hitam, 1 putih.
* Gambar punya 32 baris dan tiap baris 32 *pixel*, dibaca dari kiri atas, baris demi baris. Baris gambar 1 = `x_1` sampai `x_32`, baris 2 = `x_33` sampai `x_64`, dan seterusnya sampai baris 32 = `x_993` sampai `x_1024`.
* Satu baris gambar dicetak dalam 4 baris angka (8 *pixel* per baris). Jadi seluruh 1024 nilai tampil, bukan berhenti di `x_1017`. Itu hanya *pixel* pertama pada baris cetak terakhir.

## Sel 6. Step 4: Keluaran *Hidden Layer* (rinci)

**Rumus buku:**

```
z_in_j = v_0j + SUM_i ( x_i * v_ij )      i = 1 sampai n
z_j    = f(z_in_j)
```

**Kode (`langkah_4`)**

```python
for j in range(p):                                   # tiap node hidden, 100 kali
    z_in[j] = v_0[j] + np.sum(x * v[:, j])
    z[j]    = f(z_in[j])
```

**Cara kerja untuk satu node hidden j**
1. `v[:, j]` adalah kolom ke-j dari `v`: 1024 bobot yang menuju node hidden j (satu bobot dari tiap *pixel*).
2. `x * v[:, j]` mengalikan tiap *pixel* dengan bobotnya sendiri: `x_1 v_1j`, `x_2 v_2j`, dan seterusnya.
3. `np.sum(...)` menjumlahkan 1024 hasil itu. Ini persis `SUM_i x_i v_ij` di buku.
4. Ditambah *bias* `v_0[j]` (nilai tambahan tetap milik node hidden j), jadilah `z_in_j`.
5. `f(z_in_j)` membuat nilainya berada di 0 sampai 1, jadilah `z_j`.

Ini diulang untuk j = 1 sampai 100, sehingga hasilnya 100 nilai `z_in` dan 100 nilai `z`.

**Kenapa jumlah atas `i` memakai `np.sum`, bukan *loop* `i`?** Perulangan `i` satu per satu di Python memakan sekitar 35 detik per *epoch* hanya untuk Step 4 (hasil pengukuran saya). Hasil hitungnya sama, hanya lebih lambat.

**Contoh angka (node hidden 1)**: `z_in = +0,4708`, `z = 0,6156`. Node hidden 2: `z_in = +4,1739`, `z = 0,9848` (hampir 1, jadi node ini sangat aktif untuk gambar ini). Node hidden 3: `z_in = -7,0904`, `z = 0,0008` (hampir 0, tidak aktif).

**Cara membaca *output***: `z` dekat 1 berarti node hidden aktif, dekat 0 berarti tidak aktif.

## Sel 7. Step 5: Keluaran *Output Layer* (rinci)

**Rumus buku:**

```
y_in_k = w_0k + SUM_j ( z_j * w_jk )      j = 1 sampai p
y_k    = f(y_in_k)
```

**Kode (`langkah_5`)**

```python
for k in range(m):                                   # tiap node output, 3 kali
    y_in[k] = w_0[k] + np.sum(z * w[:, k])
    y[k]    = f(y_in[k])
```

**Cara kerja untuk satu node output k**
1. `w[:, k]` adalah 100 bobot yang menuju node output k (satu dari tiap node hidden).
2. `z * w[:, k]` mengalikan keluaran tiap node hidden dengan bobotnya, lalu `np.sum` menjumlahkannya.
3. Ditambah *bias* `w_0[k]`, jadilah `y_in_k`. Setelah `f(...)`, jadilah `y_k`.

Hasilnya tiga nilai: `y_1` (pesawat), `y_2` (mobil), `y_3` (kapal). Kelas tebakan adalah node dengan `y` terbesar.

**Contoh angka (bobot masih awal)**: `y = 0,8810` (pesawat), `0,8870` (mobil), `0,0552` (kapal). Target `t = 1 0 0`. Jaringan menebak **mobil**, padahal jawabannya **pesawat**. Wajar, karena bobotnya masih acak. Selisih inilah yang dikoreksi di Step 6 sampai 8.

## Sel 8. Step 6: Galat dan Koreksi di *Output Layer* (rinci)

**Rumus buku:**

```
delta_k    = (t_k - y_k) * f'(y_in_k)
delta_w_jk = alpha * delta_k * z_j
delta_w_0k = alpha * delta_k
```

**Kode (`langkah_6`)**

```python
for k in range(m):
    delta_k[k]   = (t[k] - y[k]) * f_aksen(y_in[k])
    delta_w_0[k] = alpha * delta_k[k]
    for j in range(p):
        delta_w[j, k] = alpha * delta_k[k] * z[j]
```

**Cara kerja**
* `t[k] - y[k]` adalah **selisih** jawaban benar dan jawaban jaringan. Positif berarti `y` kurang dari target (perlu dinaikkan), negatif berarti `y` melebihi target (perlu diturunkan).
* `f_aksen(y_in[k])` (turunan sigmoid) mengatur seberapa kuat koreksi: bila node sudah sangat dekat 0 atau 1, turunannya kecil dan koreksinya kecil.
* Hasil kali keduanya adalah `delta_k`, **faktor koreksi galat** node output k.
* `delta_w[j, k] = alpha * delta_k[k] * z[j]` adalah koreksi untuk bobot dari node hidden j ke node output k. Bobot dari node hidden yang aktif (`z_j` besar) dikoreksi lebih banyak. `alpha` (0,1) mengatur besar langkah.
* `delta_w_0[k]` adalah koreksi untuk *bias*.
* **Belum ada bobot yang berubah di sini.** Sel ini hanya menghitung seberapa besar koreksi yang diperlukan. Perubahannya baru dilakukan di Step 8.

**Contoh angka**

| k | Kelas | t | y | t - y | delta_k |
|---|-------|---|---|-------|---------|
| 1 | Pesawat | 1 | 0,8810 | +0,1190 | +0,01248 |
| 2 | Mobil | 0 | 0,8870 | -0,8870 | -0,08889 |
| 3 | Kapal | 0 | 0,0552 | -0,0552 | -0,00288 |

Mobil punya koreksi terbesar dan negatif: jaringan terlalu yakin pada mobil, jadi bobot menuju node mobil akan diturunkan. Pesawat koreksinya kecil positif: jawabannya sudah cukup dekat, hanya perlu naik sedikit.

Hitungan contoh untuk `delta_k` node 1: `(1 - 0,8810) x f'(2,0018)`, dengan `f'(2,0018) = 0,8810 x 0,1190 = 0,1048`, sehingga `0,1190 x 0,1048 = 0,01248`.

## Sel 9. Step 7: Galat dan Koreksi di *Hidden Layer* (rinci)

**Rumus buku:**

```
delta_in_j = SUM_k ( delta_k * w_jk )
delta_j    = delta_in_j * f'(z_in_j)
delta_v_ij = alpha * delta_j * x_i
delta_v_0j = alpha * delta_j
```

**Kode (`langkah_7`)**

```python
for j in range(p):
    delta_in_j[j] = np.sum(delta_k * w[j, :])
    delta_j[j]    = delta_in_j[j] * f_aksen(z_in[j])
    delta_v_0[j]  = alpha * delta_j[j]
    delta_v[:, j] = alpha * delta_j[j] * x
```

**Cara kerja**
* Node hidden tidak punya target sendiri, jadi galatnya diperoleh dari galat node output yang dikirim **mundur** (inilah arti *back propagation*).
* `delta_in_j` = jumlah `delta_k x w_jk` untuk 3 node output: seberapa besar teguran dari ketiga node output yang sampai ke node hidden j. Makin besar bobot `w_jk`, makin besar tegurannya.
* `delta_j` = `delta_in_j` dikali turunan sigmoid di node hidden j (sama seperti di Step 6).
* `delta_v[:, j] = alpha * delta_j[j] * x` memberi koreksi untuk 1024 bobot yang menuju node hidden j sekaligus. Bobot dari *pixel* yang terang (`x_i` besar) dikoreksi lebih banyak.
* `delta_v_0[j]` adalah koreksi *bias* node hidden j.
* **Catatan penting:** `w` yang dipakai di sini masih **nilai lama**, karena Step 8 baru dikerjakan sesudahnya. Buku menegaskan hal ini: semua `delta` dihitung dulu, baru bobot diperbarui.
* Bobot belum berubah di sel ini.

**Contoh angka (node hidden 1)**: `delta_in = -0,02356`, `delta_j = -0,005575`. Cek: `-0,02356 x z(1 - z)`, dengan `z = 0,6156`, sehingga `-0,02356 x 0,2366 = -0,005575`.

**Ukuran hasil**: `delta_v` berukuran 1024 x 100 (satu koreksi untuk tiap bobot V), `delta_v_0` berukuran 100.

## Sel 10. Step 8: Perbarui Bobot dan *Bias* (rinci)

**Rumus buku:**

```
w_jk(new) = w_jk(old) + delta_w_jk      w_0k(new) = w_0k(old) + delta_w_0k
v_ij(new) = v_ij(old) + delta_v_ij      v_0j(new) = v_0j(old) + delta_v_0j
```

**Kode (`langkah_8`)**

```python
for k in range(m):
    w_0[k]  = w_0[k] + delta_w_0[k]
    w[:, k] = w[:, k] + delta_w[:, k]
for j in range(p):
    v_0[j]  = v_0[j] + delta_v_0[j]
    v[:, j] = v[:, j] + delta_v[:, j]
```

**Cara kerja**
* Bobot baru = bobot lama + koreksi dari Step 6 dan 7. Fungsi ini mengubah larik `v`, `v_0`, `w`, `w_0` **langsung** (*in place*), tanpa mengembalikan nilai.
* Di sel ini fungsi dijalankan pada **salinan** bobot (`v_s`, `w_s`, dan seterusnya), supaya bobot asli tetap utuh sampai Sel 11. Pada *training* sesungguhnya, fungsi dipanggil pada bobot asli.

**Contoh angka**

| Bobot | Lama | Koreksi | Baru |
|-------|------|---------|------|
| `w_11` | +0,14900 | +0,00077 | +0,14977 |
| `w_0,1` | +0,18635 | +0,00125 | +0,18759 |
| `v_11` | +0,27396 | -0,00049 | +0,27347 |
| `v_0,1` | +0,45093 | -0,00056 | +0,45037 |

Cek `w_11`: `alpha x delta_1 x z_1 = 0,1 x 0,01248 x 0,6156 = 0,00077`. Cek `v_11`: `alpha x delta_j x x_1 = 0,1 x (-0,005575) x 0,8745 = -0,00049`.

Perubahannya kecil, dan memang begitu: satu gambar hanya menggeser bobot sedikit. Jaringan baru benar-benar belajar setelah ribuan pembaruan kecil seperti ini, dari 900 gambar dikali banyak *epoch*.

## Sel 11. Step 1, 2, dan 9: Siklus Pelatihan (rinci)

Ini sel *training* yang sesungguhnya. Isinya memanggil fungsi dari Sel 6 sampai 10, tetapi untuk **seluruh gambar latih**.

**Bagian persiapan**
* `umpan_maju(X)` menjalankan Step 4 dan 5 untuk banyak gambar tanpa mengubah bobot. Dipakai untuk menghitung akurasi.
* `akurasi(X, label)` mengambil node output terbesar (`np.argmax`) sebagai tebakan, lalu menghitung persen yang cocok dengan kelas sebenarnya.
* `riwayat_galat` dan `riwayat_akurasi` mencatat hasil tiap *epoch* untuk grafik di Sel 12.

**Inti *loop***

```python
for epoch in range(1, EPOCH_MAKS + 1):          # Step 1
    galat_total = 0.0
    for s in rng.permutation(jumlah_pola):      # Step 2
        x = X_latih[s]                          # Step 3
        t = T_latih[s]
        z_in, z = langkah_4(x, v, v_0)          # Step 4
        y_in, y = langkah_5(z, w, w_0)          # Step 5
        delta_k, delta_w, delta_w_0 = langkah_6(t, y, y_in, z)                    # Step 6
        delta_in_j, delta_j, delta_v, delta_v_0 = langkah_7(delta_k, w, z_in, x)  # Step 7
        langkah_8(v, v_0, w, w_0, delta_v, delta_v_0, delta_w, delta_w_0)         # Step 8
        galat_total += 0.5 * np.sum((t - y) ** 2)
    ...                                         # Step 9
```

**Penjelasan tiap bagian**
* **Step 1**: `for epoch ...` mengulang selama syarat berhenti belum terpenuhi. Satu putaran = satu *epoch* = seluruh 900 gambar dikerjakan sekali.
* **Step 2**: `rng.permutation(jumlah_pola)` mengacak urutan 900 gambar tiap *epoch*, lalu `for s in ...` mengambil satu gambar `s` (satu pasangan `x`, `t`) pada tiap putaran. Pengacakan mencegah jaringan terpengaruh urutan data, misalnya 300 pesawat dulu baru mobil.
* **Step 3 sampai 8**: lima baris pemanggilan fungsi di atas, urutannya sama persis dengan Sel 5 sampai 10.
* **Bobot diperbarui setelah tiap gambar**, bukan setelah semua gambar. Ini sesuai buku (hlm. 296): pembaruan per pola. Jadi dalam satu *epoch* bobot berubah 900 kali.
* `galat_total += 0.5 * np.sum((t - y) ** 2)`: galat satu gambar = setengah jumlah kuadrat selisih target dan keluaran, dijumlahkan sepanjang *epoch*.

**Step 9 (uji syarat berhenti), setelah satu *epoch* selesai**
* `galat_rata = galat_total / jumlah_pola`: rata-rata galat per gambar.
* Berhenti bila `galat_rata < AMBANG_GALAT` (0,01) **atau** `epoch == EPOCH_MAKS` (200). Bila tidak, kembali ke Step 1 untuk *epoch* berikutnya.
* Akurasi latih dihitung setelah *epoch* selesai dengan bobot terbaru, memakai `akurasi(X_latih, label_latih)`.

**Dua hal yang perlu dipahami agar tidak salah tafsir**
1. `galat_rata` dihitung dari `y` yang diperoleh **sebelum** bobot diperbarui pada tiap gambar. Jadi itu galat selama bobot masih berubah dalam *epoch* tersebut. Sementara akurasi latih dihitung sesudah *epoch* selesai. Keduanya bisa tampak tidak selalu sejalan pada *epoch* tertentu.
2. Ambang 0,01 dan batas 200 *epoch* adalah pilihan awal, belum disetel dari *training* penuh. Bisa saja jaringan berhenti karena batas 200 *epoch*, bukan karena galat sudah di bawah ambang.

**Cara membaca *output***: tabel dengan kolom *epoch*, jumlah pasangan (Step 2), galat rata-rata (Step 9), akurasi latih, dan keputusan berhenti. Tabel dicetak pada *epoch* 1, tiap 10 *epoch*, dan saat berhenti. Di *epoch* 1 juga dicetak lama satu *epoch* (sekitar 1,7 detik di lingkungan saya, di Colab bisa berbeda), sehingga 200 *epoch* kira-kira beberapa menit.

---

## Sel 12. Grafik Hasil Belajar

* Grafik kiri: galat rata-rata tiap *epoch* (diharapkan turun). Grafik kanan: akurasi data latih tiap *epoch* (diharapkan naik).
* Datanya dari `riwayat_galat` dan `riwayat_akurasi` yang dicatat di Sel 11, jadi tidak ada komputasi ulang.

## Sel 13. Evaluasi Model dan *Confusion Matrix*

* `laporan` menghitung akurasi pada data latih (900 gambar) dan data uji (150 gambar yang tidak ikut dilatih).
* Perbandingan keduanya menunjukkan *overfitting*: akurasi latih jauh di atas akurasi uji berarti jaringan menghafal data latih.
* Tabel tebakan (*confusion matrix*): baris = kelas sebenarnya, kolom = tebakan jaringan. Angka di diagonal adalah tebakan benar, di luar diagonal adalah salah tebak.

## Sel 14. Inspeksi Keluaran Jaringan terhadap Target

* Tabel pertama: pola target tiap kelas (`1 0 0`, `0 1 0`, `0 0 1`).
* Tabel kedua: nilai `Y1`, `Y2`, `Y3` untuk 3 gambar uji tiap kelas, dibandingkan dengan targetnya. Hasil BENAR bila node dengan nilai terbesar adalah node kelas sebenarnya.
* Tujuannya melihat ketegasan tebakan: node yang benar seharusnya mendekati 1 dan sisanya mendekati 0.

## Sel 15. Penyimpanan Parameter Bobot Terlatih

* `np.savez` menyimpan `v`, `v_0`, `w`, `w_0`, dan `ukuran` (lebar dan tinggi gambar) ke satu berkas `BOBOT.npz`.
* Berkas itu dibaca ulang (`np.load`) dan dibandingkan dengan bobot di memori. *Output* "Isi berkas sama dengan bobot di memori: YA" menandakan penyimpanan berhasil.
* Program pengujian (`PENGUJIAN.ipynb`) membaca berkas ini untuk mengenali satu gambar tanpa *training* ulang.
