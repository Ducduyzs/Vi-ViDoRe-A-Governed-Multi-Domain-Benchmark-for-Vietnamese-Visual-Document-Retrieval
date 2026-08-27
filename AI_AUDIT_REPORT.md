# Bao cao ban giao de kiem chung doc lap

Ngay lap: 2026-08-27

## 1. Muc tieu kiem tra

Xac minh Baseline 3 (`vidore/colpali-v1.2`) dang dung backend ColPali hop le, xu ly duoc query co do dai token khac nhau, va cac metric trong artifact co the tai lap.

Khong su dung ket qua ColPali cu sau day vi model adapter da duoc load khong dung: macro nDCG@5 `0.0738`, overall nDCG@5 `0.0705`, MRR@10 `0.0648`.

## 2. Pham vi thay doi can audit

- `src/models/visual_retriever.py`
  - Backend chinh chuyen sang `sentence_transformers.MultiVectorEncoder`.
  - Query dung `encode_query`; document dung `encode_document`.
  - Query embeddings cua cac batch duoc zero-pad den cung token length truoc khi `torch.cat`.
  - Document co token length khac nhau se bi tu choi, do zero-padding document khong an toan neu khong truyen document mask vao MaxSim.
  - `colpali_engine` va raw Transformers chi con la fallback.
- `src/models/maxsim.py`
  - `maxsim_pytorch` chuan hoa cosine, max theo document token, sau do sum theo query token.
  - Ham co ho tro `query_mask` va `doc_mask`, nhung `rank_documents_maxsim` chua truyen mask.
- `tests/test_maxsim.py`
  - Co regression test xac nhan zero-padding query khong doi MaxSim score.
- `requirements.txt`
  - `transformers>=5.15,<6.0`
  - `sentence-transformers[image]>=6.0.0,<7.0.0`
  - `colpali-engine>=0.3.0,<0.4.0`

## 3. Moi truong da chay thanh cong

- OS: Windows
- Python: `C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe` (Python 3.10)
- PyTorch: `2.6.0+cu124`
- Transformers: `5.16.1`
- Sentence Transformers: `6.0.0`
- colpali-engine: `0.3.18`
- Device: CUDA, dtype `bfloat16`
- Workspace khong phai Git repository; khong co commit SHA de xac minh provenance.

Khong dung `.venv` Python 3.13 cho lan tai lap nay.

## 4. Lenh tai lap

Test tap trung:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' -m pytest tests/test_maxsim.py -q
```

Ket qua mong doi:

```text
5 passed
```

Benchmark:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe' scripts/03_run_baselines.py --split test --skip_biencoder --run_visual --visual_model vidore/colpali-v1.2
```

Du lieu lan chay: 90 queries, 120 corpus pages. Document embeddings co shape `(120, 1030, 128)`; smoke test query/document co shape `(1, 20, 128)` va `(1, 1030, 128)`.

## 5. Ket qua can doi chieu

| Model | Macro nDCG@5 | Overall nDCG@5 (95% CI) | MRR@10 (95% CI) |
|---|---:|---:|---:|
| Native Text + BM25 | 0.5358 | 0.6350 [0.5565, 0.7148] | 0.5967 [0.5223, 0.6745] |
| ColPali Zero-Shot | 0.5175 | 0.6775 [0.5957, 0.7621] | 0.6496 [0.5602, 0.7317] |

Chenh lech ColPali tru BM25:

- Macro nDCG@5: `-0.0183`
- Overall nDCG@5: `+0.0425`
- MRR@10: `+0.0529`
- Education nDCG@5: `+0.0831` (`0.7841` so voi `0.7010`)
- Legal nDCG@5: `-0.1197` (`0.2510` so voi `0.3707`)

Khoang tin cay bi chong lap; khong ket luan y nghia thong ke chi tu bang nay.

Artifact chinh: `results/benchmark_test_results.json`

SHA-256:

```text
1D5A166173D46008F79C11D56696090A2CBFAD2F4CDCC6E1F219C726EFE9C57B
```

Parser doc lap da doc JSON va thu duoc:

```text
Native Text + BM25: 90, 0.5358, 0.6350, 0.5967
ColPali Zero-Shot:   90, 0.5175, 0.6775, 0.6496
```

## 6. Diem reviewer phai kiem tra ky

1. Khi load model, Transformers van in `UNEXPECTED` cho `custom_text_proj.weight/bias` va `custom_text_proj.lora_A/lora_B`. Co the day la qua trinh load tung submodule roi assembly cua `MultiVectorEncoder`, nhung chua co bang chung truc tiep trong repo rang LoRA da duoc gan dung. Reviewer nen kiem tra parameter names, checksum/summary cua adapter, hoac so sanh embedding voi API tham chieu chinh thuc.
2. Checkpoint khong pin Hugging Face revision; cung model ID co the thay doi theo thoi gian. Requirements cung la range, khong phai lockfile.
3. Bao cao Markdown hien thi `scanned nDCG@5 = 0.0`, trong khi JSON chi co nhom `born_digital`. Day la nhom vang mat, khong nen dien giai la hieu nang tren scanned documents bang 0.
4. Test split chi gom domain `education` va `legal`; cot financial/health rong. Macro-domain vi vay la trung binh tren hai domain co mat.
5. Lan chay dung `--skip_biencoder`, nen artifact hien tai chi co BM25 va ColPali; khong phai bang day du ba baseline.
6. Script bat exception cua visual baseline va van co the ghi report. Reviewer phai xac nhan log co dong `Corpus indexed`, `Running MaxSim retrieval`, va metric ColPali, khong chi dua vao exit code 0.
7. Zero-padding query an toan trong implementation hien tai vi vector 0 sau normalize van la 0 va dong gop 0 vao tong. Lap luan nay khong ap dung cho document padding: token 0 co the thang cac cosine am trong phep max.

## 7. Tieu chi chap nhan

- Focused tests dat `5 passed`.
- Backend runtime la `sentence_transformers`.
- Khong xuat hien fallback sang `colpali_engine` hoac raw Transformers.
- Query/document embedding dimension cuoi la 128; document token length la 1030 cho corpus hien tai.
- JSON duoc parse thanh cong va metric khop muc 5.
- Reviewer giai quyet hoac danh dau chap nhan rui ro cua canh bao `custom_text_proj`.
- Khong bao cao ket qua cu `0.0738/0.0705/0.0648`.

## 8. File lien quan

- `src/models/visual_retriever.py`
- `src/models/maxsim.py`
- `tests/test_maxsim.py`
- `scripts/03_run_baselines.py`
- `requirements.txt`
- `results/benchmark_test_results.json`
- `results/benchmark_test_results.md`
- `results/benchmark_test_results.tex`
