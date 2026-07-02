# Workflow-CI

Repository ini berisi MLflow Project dan GitHub Actions CI/CD pipeline untuk
re-training otomatis model klasifikasi Facies dan publikasi Docker image,
sebagai bagian dari Kriteria 3 untuk Proyek Akhir Kelas Membangun Sistem Machine Learning (MSML) — Dicoding.

Environment dikelola menggunakan **[uv](https://docs.astral.sh/uv/)**.

Lihat hasil Kriteria 1 di link berikut:
https://github.com/wapratama/Eksperimen_SML_Wisnu-anugrah-pratama

## Struktur Repository

```
Workflow-CI/
├── .github/workflows/ci.yml       # GitHub Actions: retrain + build + push
├── MLProject/
│   ├── MLProject                  # Spesifikasi MLflow Project (tanpa conda_env)
│   ├── modelling.py               # Script training (adaptasi dari Kriteria 2)
│   ├── pyproject.toml             # Dependency spec (uv)
│   ├── uv.lock                    # Lockfile
│   └── facies_preprocessing.csv   # Dataset hasil Kriteria 1
├── Docker_Hub.txt                 # Link Docker Hub (gitignored)
└── README.md
```

---

## Panduan

### Penjelasan — Kenapa Tidak Ada `conda.yaml`?

Repo ini merupakan eksperimen dengan tidak menyertakan environment spec apa pun yang menggunakan
`conda_env: conda.yaml` di file `MLProject` sesuai dengan tutorial kelas. Sebagai gantinya:

```
MLProject (spec file):
  command: "uv run python modelling.py ..."

Dijalankan dengan flag:
  mlflow run . --env-manager=local
```

`--env-manager=local` memberi tahu MLflow: *"jangan buat environment apa
pun, jalankan command persis apa adanya di environment yang sedang
aktif"*. 

Karena command-nya sendiri diawali `uv run`, `uv` yang mengurus
pembacaan `pyproject.toml`/`uv.lock` dan eksekusi di virtual environment
yang sesuai, MLflow hanya jadi orchestrator run tracking, bukan pengelola dependency.

**Hasil pengujian** (dijalankan di environment testing):
```
=== Running command 'uv run python modelling.py --data_path facies_preprocessing.csv
     --n_estimators 200 --max_depth 25' in run with ID '297ed46da34b468399d465de6b68f57a' ===
Test accuracy : 0.8253
Test f1_macro : 0.8309
=== Run (ID '297ed46da34b468399d465de6b68f57a') succeeded ===
```

---

### Prasyarat GitHub Secrets

Sebelum push, tambahkan di **Settings > Secrets and variables > Actions**:

| Secret Name | Nilai |
|---|---|
| `DOCKERHUB_USERNAME` | Username Docker Hub kamu |
| `DOCKERHUB_TOKEN` | Access Token (bukan password) dari Docker Hub |

Cara membuat Access Token: Docker Hub → Account Settings → Security →
New Access Token.

Docker Hub → Account Settings → Personal access tokens →
Generate new token.

---

### Cara Kerja Workflow CI (13 Steps)

Workflow `ci.yml` berjalan otomatis setiap push ke `MLProject/`, dengan tahapan:

| # | Step | Fungsi |
|---|---|---|
| 1 | Checkout | Clone repo ke runner |
| 2-3 | Install uv + Python 3.12 | Siapkan tool via `astral-sh/setup-uv` |
| 4 | Check Env | Verifikasi versi (debug aid) |
| 5 | Install dependencies | `uv sync --locked` di dalam `MLProject/` |
| 6 | Run mlflow project | `mlflow run` re-training model |
| 7 | Get run_id | Parse output untuk ID run terbaru |
| 8 | Upload to GitHub | Commit `mlruns/` kembali ke repo |
| 9 | Build Docker Model | `mlflow models build-docker` |
| 10 | Login Docker Hub | Autentikasi pakai secrets |
| 11 | Tag Image | Beri tag `latest` + `<run_id>` |
| 12 | Push Image | Kirim ke Docker Hub |
| 13 | Complete job | Log ringkasan |

---

### Menjalankan Secara Lokal (untuk Verifikasi)

```bash
cd MLProject
uv sync

uv run mlflow run . --env-manager=local \
    --experiment-name "Facies Classification - CI Retraining" \
    -P data_path=facies_preprocessing.csv \
    -P n_estimators=200 \
    -P max_depth=25
```

Catatan: Jika sukses di lokal, kemungkinan
besar juga akan sukses di GitHub Actions.

---