# Danh gia muc do san sang cong bo

Ngay audit: 2026-08-27

## Ket luan dieu hanh

Du an **chua san sang nop bai**. Ket qua hien tai la pilot ky thuat, khong phai ket qua benchmark human-validated.

- **Huong 2 (Vi-ViDoRe):** nen la huong uu tien. Da co governed candidate, policy, annotation guideline, split lock va freeze gates. Tuy nhien benchmark dang `BLOCKED` boi 8 gates.
- **Huong 1 (AdaBudget-Col):** moi co exact MaxSim/ColPali baseline. Chua co token pooling, pruning, adaptive controller, compression, latency hay Pareto experiment; chua du bang chung cho claim thuat toan.
- Metric pilot ColPali `0.5175` macro nDCG@5, `0.6775` overall nDCG@5 va `0.6496` MRR@10 chi nen dung de debug pipeline. Khong dua vao abstract/main table cua paper.

## 1. Tien do co bang chung

### Huong 2: benchmark Vietnamese VDR

Da co:

- Governed candidate: 15 documents, 369 pages, 956 queries tren train/dev/test.
- Test candidate: 8 documents, 299 pages, 888 queries, 4 domain.
- Zero exact duplicate va zero source/template group leakage theo registry hien tai.
- Data governance policy, annotation guideline, candidate split lock va automated freeze report.
- BM25 va ColPali zero-shot pipeline; ColPali dung `MultiVectorEncoder` va exact MaxSim.
- 22 tests cua repository: tai thoi diem audit, governance test chay lai sau rebuild dat `5/5`; full suite truoc rebuild co `21 passed, 1 failed` do artifact audit cu thieu gate, sau do builder da cap nhat artifact.

Chua co:

- Frozen benchmark manifest.
- Human-written core va final human judgments.
- Complete pooled qrels; labels hien tai chi la automatically generated target pages.
- Corpus quy mo publication target, baseline suite, significance test, efficiency benchmark va error analysis.

### Huong 1: adaptive token budgeting

Da co:

- ColPali full-token encoding va exact cosine MaxSim.
- Regression test cho variable-length query batches.
- Pilot full-token baseline tren corpus nho.

Chua co toan bo dong gop trung tam:

- Hierarchical/static pooling reproduction.
- Fixed-K salience pruning va merge baselines.
- Dynamic budget controller va budget constraint.
- Quality-storage-latency sweeps, Pareto frontier/hypervolume.
- Cross-domain/cross-backbone validation, ablation va large-corpus system benchmark.

Vi vay Huong 1 chua vuot moc Week 1-2 trong research plan.

## 2. Freeze blockers cua Huong 2

Bao cao `data/benchmark_governed_v0_1/FREEZE_BLOCKED.md` hien co 8 blocker:

| Uu tien | Blocker | Trang thai/thieu hut | Cach dong gate |
|---|---|---|---|
| P0 | Final human judgments | 0 annotation rows; can 2 independent judgments/pair | Pool candidates, double annotate, adjudicate, agreement >= 0.67 |
| P0 | Human-written ratio | 0%; yeu cau >= 40% | Viet/sua query bang nguoi, luu provenance chi tiet |
| P0 | License | 6 test PDFs chua verified | Xac minh quyen hoac thay/loai tai lieu |
| P0 | Complete documents | 6 test PDFs chi co `first_10` | Xu ly full PDF hoac loai khoi test |
| P0 | Query/domain coverage | education 11, legal 15; can >= 50/domain | Them it nhat 39 education va 35 legal query hop le |
| P0 | Scanned targets | 0; can >= 20 | Human workflow tao va judge >= 20 query nham scan |
| P0 | Page-type targets | table 4, chart 2, form 0; can >= 20 moi loai | Bo sung toi thieu 16 table, 18 chart, 20 form queries |
| P0 | Source diversity | legal 1 source, finance 1 source; can >= 2/domain | Them nguon doc lap, tranh domain-publisher confound |

Luu y: test co 30 scanned pages va 35 chart-heavy pages, nhung query khong nham vao chung. Day la loi coverage cua query/judgment, khong chi la loi corpus.

## 3. Cac luan diem can cung co

### Claim A: "Can mot benchmark VDR tieng Viet"

Can them systematic literature review co cutoff date, search strings, databases, inclusion/exclusion table. Khong dung "first" neu chua co protocol tim kiem tai lap.

### Claim B: "Benchmark dai dien cho tai lieu Viet"

Hien chua bao ve duoc: 15 documents la qua nho; test bi lech healthcare `699/888` queries; financial/legal moi co mot source; khong co infographic domain; table/chart/form targets thieu. Can tang documents, source diversity va bao sample counts kem confidence intervals cho moi slice.

### Claim C: "Human-validated va relevance day du"

Hien sai neu tuyen bo. Query chu yeu la heuristic fallback (`859/888` test), khong co annotation. Can pooling tu BM25 + Vietnamese dense + visual, graded `0/1/2`, double annotation, adjudication, agreement va audit unjudged documents. Mot target page/query khong phai qrels day du.

### Claim D: "Visual late interaction tot hon OCR/text retrieval"

Pilot chi so ColPali voi native-text BM25. Can OCR+BM25, OCR+dense Vietnamese/BGE-M3, native-text strong baseline, multilingual image single-vector, modern ColVision checkpoint va paired significance. Phai phan tich born-digital/scan va OCR-quality slices; khong duoc dung OCR yeu lam strawman.

### Claim E: "ColPali cai thien retrieval"

Pilot cho overall nDCG@5 tang `+0.0425` va MRR@10 tang `+0.0529` so voi BM25, nhung macro nDCG@5 giam `-0.0183`, legal giam `-0.1197`, CI chong lap. Chi co the noi day la tin hieu de nghien cuu, chua phai cai thien co y nghia thong ke.

### Claim F: "Vietnamese adaptation hieu qua"

Chua co model adapted. Train/dev hien chi co education, nen khong the tach language adaptation khoi domain transfer. Can train data da domain, hard negatives, zero-shot vs LoRA vs curriculum, >=3 seeds va retention tren ViDoRe goc.

### Claim G: "Adaptive budgeting moi va hieu qua"

Chua co evidence. Phai thang static/hierarchical pooling o cung vectors, bytes hoac latency; bao quality drop, worst group, overhead controller va wall-clock P95. Giam FLOPs khong tu dong la speedup.

## 4. Lo hong code/protocol can va

### P0: anh huong tinh hop le cua ket qua

1. Pilot `qrels.tsv` sinh tu source target duy nhat; false negatives chua duoc pool/judge. Metric pilot khong phai estimate benchmark hop le.
2. Script baseline bat exception ColPali roi van ghi report va co the exit 0. Paper pipeline can fail-fast neu baseline bat buoc that bai.
3. Model ID khong pin Hugging Face revision; requirements la range, khong co lockfile/commit SHA. Can manifest moi truong va checkpoint revision.
4. Loader con canh bao `UNEXPECTED custom_text_proj`/LoRA. Can kiem chung parameter loading bang API tham chieu, parameter inventory/checksum hoac embedding parity test.
5. Test set candidate da duoc chay nhieu lan. Sau khi qrels human-final duoc freeze, can tach blind final test hoac co mot held-out evaluation set moi de tranh adaptive overfitting.

### P1: anh huong bao cao va tai lap

1. Evaluator chi co bootstrap CI rieng tung model; can paired bootstrap/randomization test tren per-query deltas va multiple-comparison correction.
2. Bootstrap hien khong co RNG seed, nen CI khong deterministic.
3. Report Markdown/LaTeX dien subgroup vang mat thanh `0.0`, tao an tuong sai rang model dat 0 thay vi khong co mau. Phai hien `N/A` va sample count.
4. Evaluator suy source type tu target page dau tien; query co nhieu relevant page/source type co the bi gan slice sai.
5. Report khong luu per-query rankings/scores, runtime config, hardware, seed va model revision. Can artifact manifest cho moi run.
6. Query generation nuot exception (`except: pass`), co the gay selection bias am tham. Can error log, failure counts va non-zero status neu vuot threshold.
7. Near-duplicate/template audit can phat hanh chi tiet pHash/text similarity; registry group pass chua du de chung minh contamination-free.

### P2: engineering va presentation

1. `--run_visual` dung `store_true` voi default `True`, nen khong co cach CLI de tat visual baseline.
2. README chua mo ta setup, data status, governed-vs-legacy distinction, reproducibility commands va known limitations.
3. Workspace khong phai Git repository, nen khong co immutable code revision de citation.

## 5. Lo trinh ngan nhat toi bai co the nop

### Milestone 1: khoa data governance

1. Giai quyet license/full-document cho 6 PDFs; them source legal va finance.
2. Bo sung form/table/chart/scan pages co license ro rang.
3. Can bang query theo domain; khong tang healthcare them.
4. Chay builder den khi chi con annotation gate.

### Milestone 2: human benchmark

1. Pilot 100 queries, double annotation, sua guideline neu agreement < 0.67.
2. Human-write/edit >= 40%; tach `human_written`, `llm_edited`, `llm_accepted`, `heuristic`.
3. Pool top-k tu it nhat BM25, dense Vietnamese va ColPali/ColQwen.
4. Double-judge moi pooled pair, adjudicate, dong tat ca freeze gates.

### Milestone 3: credible experiments

1. Chay baseline suite toi thieu tren frozen benchmark.
2. Luu per-query runs; paired significance + subgroup sample counts + error analysis.
3. Do P50/P95 latency, indexing throughput, index size, VRAM va hardware.
4. Chi sau do fine-tune mot backbone; bao >=3 seeds va ViDoRe retention.

### Milestone 4: paper package

1. Dataset/model card, provenance/license table, contamination report, annotation agreement.
2. Lockfile/container, Git commit, checkpoint revisions va run manifests.
3. Main table chi lay frozen benchmark; pilot result vao appendix voi nhan `pipeline validation` neu can.

## 6. Go/no-go

- **Nop resource/benchmark paper:** GO chi khi tat ca freeze gates pass va co baseline/significance/error analysis.
- **Nop model paper:** NO-GO den khi co adapted model, >=3 seeds, strong baselines va out-of-domain retention.
- **Nop adaptive-token paper:** NO-GO den khi adaptive method khong bi static pooling dominate tren Pareto frontier.
- **Workshop/demo hien tai:** co the trinh bay nhu work-in-progress/pipeline, nhung khong goi la human-validated benchmark va khong claim state of the art.

## 7. Danh gia tong quan

Nen tap trung Huong 2. Governance architecture da dat nen mong dung; bottleneck hien tai la data rights va human annotation, khong phai them model. Neu giai quyet P0 truoc, du an co mot duong cong bo ro rang. Neu tiep tuc fine-tune/pruning tren candidate hien tai, ket qua se kho bao ve truoc reviewer vi qrels va test composition chua hop le.