# Klasifikasi Multiclass Pneumonia Menggunakan MobileNetV2 (X-Ray Classification)

Proyek ini merupakan implementasi deep learning menggunakan model **MobileNetV2** untuk melakukan **klasifikasi citra X-ray paru-paru** ke dalam tiga kelas:
-  **Normal**
-  **Pneumonia Bakterial**
-  **Pneumonia Viral**

Notebook ini dikembangkan dengan **Python di Google Colab**, memanfaatkan dataset citra X-ray dari pasien yang tersedia dalam format ZIP dan diekstrak langsung dari Google Drive.

---

## Tujuan Proyek

- Mendeteksi dan mengklasifikasikan jenis pneumonia menggunakan citra X-ray.
- Melakukan pemisahan label pneumonia (BACTERIAL dan VIRAL) berdasarkan nama file.
- Melatih model MobileNetV2 dengan teknik transfer learning.
- Mengevaluasi model dengan confusion matrix dan classification report.

---

## Tools & Library yang Digunakan

| Library / Tool       | Keterangan                                  |
|----------------------|----------------------------------------------|
| TensorFlow / Keras   | Framework untuk deep learning dan CNN        |
| MobileNetV2          | Pretrained model ringan untuk transfer learning |
| NumPy, Pandas        | Manipulasi data                              |
| Matplotlib, Seaborn  | Visualisasi grafik dan evaluasi              |
| Scikit-learn         | Confusion matrix & classification report     |
| Google Colab         | Lingkungan pengembangan                      |

---

## Struktur Dataset

Dataset awal tersedia dalam format ZIP dan diakses dari Google Drive. Setelah diekstrak, struktur dataset dibuat ulang untuk mengelompokkan gambar `PNEUMONIA` ke dalam dua kelas:

| Kelas        | Deskripsi                                       |
|--------------|-------------------------------------------------|
| `NORMAL`     | Gambar X-ray pasien sehat                       |
| `BACTERIAL`  | Gambar pneumonia bakteri (filename mengandung "bacteria") |
| `VIRAL`      | Gambar pneumonia virus (filename mengandung "virus")    |


Dataset yang digunakan adalah **Chest X-ray Images (Pneumonia)**, terdiri dari data pelatihan (`train`), pengujian (`test`), dan validasi (`val`) dengan masing-masing subfolder kelas.

| Folder        | Subfolder Kelas  | Deskripsi                                     |
|---------------|------------------|-----------------------------------------------|
| `train/`      | `NORMAL/`        | Citra X-ray pasien tanpa pneumonia            |
|               | `BACTERIAL/`     | Citra X-ray dengan pneumonia bakteri          |
|               | `VIRAL/`         | Citra X-ray dengan pneumonia virus            |

| `test/`       | `NORMAL/`        | Data uji untuk mengukur performa model        |
|               | `BACTERIAL/`     |                                               |
|               | `VIRAL/`         |                                               |

| `val/`        | `NORMAL/`        | Data validasi saat proses training            |
|               | `BACTERIAL/`     |                                               |
|               | `VIRAL/`         |                                               |

---

##  Langkah-langkah Proyek

1. **Import library** dan mount Google Drive.
2. **Ekstrak file ZIP dataset** ke direktori kerja Colab.
3. **Pemisahan label pneumonia** berdasarkan kata kunci `'bacteria'` dan `'virus'`.
4. **Data Augmentation** menggunakan `ImageDataGenerator`.
5. **Build dan fine-tune model MobileNetV2**:
   - Global Average Pooling
   - Dropout
   - Dense + Softmax (3 kelas)
6. **Compile dan train model** dengan Adam optimizer.
7. **Evaluasi performa model** dengan confusion matrix dan classification report.

---

## Evaluasi & Output

- **Akurasi model** selama training & validation ditampilkan dalam grafik.
- **Confusion matrix** menunjukkan distribusi prediksi model.
- **Classification report** mencakup precision, recall, dan f1-score untuk masing-masing kelas.
- **Visualisasi hasil**: prediksi benar dan salah ditampilkan untuk analisis.

---

##  Catatan Tambahan

- Model ini menggunakan transfer learning sehingga lebih efisien dan ringan.
- Struktur dataset harus konsisten agar generator dapat membaca label dengan benar.
- File pneumonia awal tidak langsung terbagi menjadi dua kelas, pemisahan dilakukan dengan `if 'bacteria' in filename`.

---

##  Bonus Point A – Uji Beberapa Model

Saya telah menguji beberapa arsitektur CNN untuk mengetahui mana yang paling optimal untuk tugas klasifikasi pneumonia 3 kelas ini.

###  Perbandingan Model

| Model          | Akurasi Validasi | Waktu Training | Catatan                          |
|----------------|------------------|----------------|----------------------------------|
| **MobileNetV2** | ~80%            | Cepat          | Akurat, ringan, efisien          |
| EfficientNetB0  | ~48%            | Sedang         | Belum optimal, perlu fine-tuning |


**Jadi**:  
- MobileNetV2 menunjukkan performa terbaik dengan akurasi validasi tertinggi (~80%) dan waktu pelatihan tercepat.
- EfficientNetB0 tidak bekerja optimal pada dataset ini.
- MobileNetV2 dipilih sebagai **model akhir** untuk klasifikasi pneumonia 3 kelas.

---


> Saya percaya bahwa teknologi cerdas dapat membantu deteksi dini penyakit dan menyelamatkan lebih banyak nyawa secara efisien.

---
