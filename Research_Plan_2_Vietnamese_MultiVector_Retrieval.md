# Đề cương Nghiên cứu 2: Vietnamese Visual Document Retrieval

> **Tên dự kiến:** *Vi-ViDoRe: A Human-Validated Benchmark for Vietnamese Visual Document Retrieval*  
> **Tên mở rộng mô hình:** *Adapting Multimodal Late-Interaction Retrievers to Complex Vietnamese Documents*  
> **Mục tiêu phù hợp:** ACL/AACL Findings, CIKM resource/short paper, PACLIC, RIVF, KSE, IEEE Access.  
> **Thời gian:** 14--20 tuần tùy nguồn lực gán nhãn.

---

## 1. Tóm tắt và đánh giá

Tài liệu tiếng Việt có dấu, font hành chính, bảng biểu, nhiều cột, scan và con dấu tạo ra một bài toán visual document retrieval có giá trị rõ ràng. Đóng góp mạnh nhất không phải chỉ là fine-tune một checkpoint mới, mà là **benchmark tiếng Việt có protocol chống leakage, query tự nhiên, hard negatives và đánh giá human-validated**.

| Tiêu chí | Điểm | Nhận xét |
| :--- | :---: | :--- |
| Tính mới benchmark | 8/10 | Mạnh nếu literature review xác nhận khoảng trống và test set có con người viết/thẩm định. |
| Tính mới mô hình | 6/10 | Fine-tune ColQwen trên dữ liệu Việt là incremental; cần data strategy hoặc analysis rõ ràng. |
| Tác động | 9/10 | Hữu ích cho RAG pháp luật, tài chính, y tế và hành chính. |
| Khả thi 3--5 tháng | 7/10 | Benchmark-first khả thi; benchmark + hai backbone + 10 baseline đầy đủ là quá tải. |
| Rủi ro chính | Cao | Bản quyền/PII, LLM artifacts, false negatives, trùng PDF/template và chi phí thẩm định. |

**Khuyến nghị:** ưu tiên bài **Vi-ViDoRe benchmark + benchmarked adaptation of one backbone**. Không phát triển đồng thời Vi-ColPali và Vi-ColQwen. Chỉ dùng từ “đầu tiên” sau systematic search có ngày cutoff.

---

## 2. Bối cảnh và khoảng trống

### 2.1. Thách thức thực tiễn

- OCR có thể sai dấu, số liệu, thứ tự đọc và cấu trúc bảng.
- Single-vector pooling khó giữ nhiều tín hiệu nhỏ trên một trang.
- Tài liệu Việt có domain language, viết tắt, điều/khoản/điểm và font/scan đặc thù.
- Retrieval benchmark cần đánh giá tìm **đúng trang trong corpus**, không chỉ trả lời câu hỏi khi đã biết trang.

### 2.2. Hiệu chỉnh tuyên bố kỹ thuật

- ViDoRe gốc đã gồm tiếng Anh và tiếng Pháp; đóng góp ở đây là độ phủ **tiếng Việt và miền tài liệu Việt**, không phải benchmark VDR đa ngữ đầu tiên.
- BGE-M3 hỗ trợ dense/sparse/multi-vector cho **văn bản**. Trong đề tài này, nó là baseline sau PDF parsing/OCR, không phải visual late-interaction model.
- Text tokenizer không nên tách tùy ý khỏi VLM nếu làm mất shared embedding space. Phương án an toàn là tokenizer gốc của backbone, có ablation vocabulary expansion hoặc adapter nếu thật sự cần.
- “Pre-training on Vietnamese documents” chỉ đúng nếu có quy mô và objective tiền huấn luyện phù hợp. Với LoRA trên cặp query--page, thuật ngữ đúng là **retrieval adaptation/fine-tuning**.

Khoảng trống cần kiểm chứng: chưa có benchmark công khai đủ rộng cho page-level visual retrieval tiếng Việt, có document-level split, human validation, hard-negative corpus và license/ethics rõ ràng.

---

## 3. Câu hỏi nghiên cứu và giả thuyết

- **RQ1:** Visual late interaction có vượt OCR-based retrieval trên tài liệu Việt không?  
	**H1:** lợi ích lớn nhất xuất hiện ở table/chart/scan, nhỏ hơn ở born-digital text-heavy.
- **RQ2:** Năng lực zero-shot của retriever đa ngữ hiện tại yếu ở đâu?  
	**H2:** lỗi tập trung ở số liệu có dấu phân cách Việt, điều/khoản, font scan và bảng nhiều cấp.
- **RQ3:** Vietnamese adaptation cải thiện do ngôn ngữ, domain hay visual style?  
	**H3:** training phối hợp paraphrase tiếng Việt + hard negatives cùng PDF tốt hơn chỉ dịch dữ liệu tiếng Anh.
- **RQ4:** Cải thiện có tổng quát ngoài nguồn/domain huấn luyện không?  
	**H4:** model tăng macro nDCG@5 trên source-held-out test mà không giảm đáng kể trên ViDoRe gốc.
- **RQ5:** LLM-generated queries có đánh giá giống truy vấn người dùng không?  
	**H5:** ranking giữa các model khác nhau đáng kể giữa synthetic-only và human-written subsets; vì vậy phải báo riêng.

---

## 4. Phạm vi benchmark Vi-ViDoRe

### 4.1. Đơn vị truy xuất và quy mô

- Đơn vị corpus: **trang tài liệu**; lưu liên kết `document_id`, `page_number`, nguồn và hash.
- Corpus mục tiêu: tối thiểu 10.000 trang test-candidate, ưu tiên 20.000+ nếu storage/annotation cho phép.
- Test queries: 2.000--3.000; ít nhất 40% human-written, phần còn lại LLM-assisted nhưng human-validated.
- Dev queries: 500--800 từ document/source khác test.
- Training: 20.000--50.000 weakly supervised query--page pairs, tách hoàn toàn khỏi dev/test theo PDF, nguồn và template cluster.

Không dùng 3.000 QA pairs vừa để train model vừa để tuyên bố benchmark. Training corpus và evaluation benchmark là hai artifact riêng.

### 4.2. Miền dữ liệu

| Domain | Dạng trang ưu tiên | Ví dụ query |
| :--- | :--- | :--- |
| Pháp luật/hành chính | Điều-khoản-điểm, biểu mẫu, phụ lục | Tìm trang quy định mức phạt và điều kiện áp dụng. |
| Tài chính/doanh nghiệp | Bảng cân đối, thuyết minh, chart | Tìm trang chứa chỉ tiêu và kỳ báo cáo cụ thể. |
| Y tế/dược | Hướng dẫn, liều dùng, bảng chống chỉ định | Tìm trang nêu liều theo nhóm tuổi. |
| Giáo dục/khoa học | Giáo trình, công thức, sơ đồ | Tìm trang giải thích công thức hoặc hình. |
| Infographic/thống kê | Chart, bản đồ, chú giải | Tìm biểu đồ so sánh tỉnh hoặc năm. |

Mỗi domain cần cả born-digital và scanned; tránh để domain đồng nghĩa với một source hoặc một visual style.

### 4.3. Taxonomy truy vấn

- Lexical/entity lookup.
- Điều/khoản và cross-reference.
- Numeric/table lookup.
- Multi-cell comparison hoặc arithmetic nhẹ.
- Chart/legend interpretation.
- Layout-dependent query.
- Paraphrase, viết tắt và query không chứa exact answer string.

Mỗi query có metadata: domain, page type, scan quality, reasoning type, answerability, nguồn tạo (human/LLM-assisted), độ khó và số trang relevant.

### 4.4. Relevance judgments

Một cặp query--gold page không đủ vì corpus có thể chứa nhiều trang trả lời đúng. Quy trình:

1. Lấy pooled top-$k$ từ BM25, dense OCR, ColPali/ColQwen và model đề xuất.
2. Annotator chấm `0 = not relevant`, `1 = partially relevant`, `2 = fully relevant`.
3. Adjudication cho bất đồng và mẫu kiểm tra chất lượng.
4. Báo tỷ lệ agreement: Cohen's kappa cho hai người hoặc Krippendorff's alpha khi nhiều người.

Nên double-annotate toàn bộ test nếu ngân sách cho phép; tối thiểu double-annotate 20--30% và mọi query có bất đồng/hard case.

---

## 5. Thu thập, gán nhãn và quản trị dữ liệu

### 5.1. Thu thập và provenance

- Chỉ lấy tài liệu công khai có điều khoản cho phép nghiên cứu/phân phối; lưu URL, publisher, ngày truy cập, license và checksum.
- Không mặc định “công khai trên web” đồng nghĩa được phép redistributing PDF/image.
- Với nguồn hạn chế, phát hành URL + hash + script dựng lại thay vì ảnh gốc.
- Loại PII, bệnh án thật, chữ ký cá nhân, số tài khoản và tài liệu có thông tin nhạy cảm.

### 5.2. Render và quality control

- Lưu PDF gốc; render theo cạnh dài/pixel budget phù hợp processor thay vì bắt buộc 300 DPI cho mọi model.
- Kiểm tra rotation, blank page, corruption, duplicate và near-duplicate.
- Dùng SHA-256 cho exact duplicate; perceptual hash + OCR/text similarity cho near-duplicate/template.
- Gắn nhãn born-digital/scan, DPI ước lượng, blur, skew, contrast và OCR confidence.

### 5.3. Sinh query

- **Human-written core:** annotator nhìn trang trong ngữ cảnh tài liệu và viết query giống nhu cầu tìm kiếm, không nhắc “trang này/hình trên”.
- **LLM-assisted:** VLM đề xuất query/answer/evidence; người sửa hoặc loại, không chỉ bấm accept.
- Query phải tự đủ nghĩa nhưng không chứa thông tin định danh trang quá dễ như tiêu đề đầy đủ hiếm gặp.
- Hard negatives lấy từ cùng PDF, cùng domain, cùng template và lexical nearest neighbors.

### 5.4. Chống contamination

Split theo thứ tự ưu tiên: publisher/source-held-out, PDF-held-out, template-cluster-held-out. Không split ngẫu nhiên theo trang.

Kiểm tra:

- exact/near duplicate giữa train--dev--test;
- cùng phiên bản văn bản hoặc báo cáo năm kế tiếp dùng chung template;
- query paraphrase/translation overlap;
- overlap với dữ liệu public đã dùng để fine-tune checkpoint nếu thông tin này có thể xác định.

Phát hành contamination report và danh sách hash, không chỉ mô tả bằng lời.

### 5.5. Dataset card

Dataset card cần có: mục đích, nguồn, license, schema, split, annotation guideline, annotator profile/compensation, agreement, PII process, known biases, limitations, takedown process và citation.

---

## 6. Mô hình và chiến lược huấn luyện

### 6.1. Backbone

Chọn **một** backbone chính sau pilot: checkpoint ColQwen/ColQwen2.5 nhỏ nhất đáp ứng quality và GPU budget, hoặc ColPali nếu cần reproduction dễ hơn. Khóa model revision và `colpali-engine` commit.

Không tuyên bố Qwen đọc tiếng Việt tốt chỉ dựa trên khả năng sinh văn bản; phải đo zero-shot retrieval trước.

### 6.2. Ba mức adaptation

1. **Zero-shot:** checkpoint gốc, không sửa model.
2. **LoRA retrieval adaptation:** in-batch contrastive MaxSim trên query--positive page, thêm mined negatives.
3. **Curriculum hard-negative tuning:** lexical negatives, same-domain negatives, same-PDF negatives và teacher-mined false positives.

Objective cơ bản:

$$s(q,d)=\sum_{i=1}^{N_q}\max_{j=1}^{N_d}\langle E_q^{(i)},E_d^{(j)}\rangle,$$

$$\mathcal{L}_{\text{InfoNCE}}=-\log\frac{\exp(s(q,d^+)/T)}{\exp(s(q,d^+)/T)+\sum_{d^-}\exp(s(q,d^-)/T)}.$$

Theo dõi false negatives trong cùng PDF; không coi mọi page khác trong batch là negative nếu chưa kiểm tra relevance.

### 6.3. Vietnamese-specific ablation

- Synthetic Vietnamese only vs translated English + native Vietnamese.
- Random negatives vs same-domain vs same-PDF hard negatives.
- Frozen backbone + projector vs LoRA language layers vs LoRA multimodal layers.
- Query augmentation tokens on/off.
- Tokenizer gốc vs vocabulary expansion chỉ khi tokenizer analysis chứng minh fragmentation bất thường.
- Born-digital only vs thêm scan augmentation.

---

## 7. Baseline và protocol đánh giá

### 7.1. Baseline tối thiểu

**OCR/parsing text retrieval**

1. PaddleOCR hoặc engine Việt được khóa phiên bản + BM25.
2. OCR + `bkai-foundation-models/vietnamese-bi-encoder` hoặc baseline Việt mạnh được xác nhận tại thời điểm chạy.
3. OCR + BGE-M3 dense, sparse và multi-vector/ColBERT mode nếu implementation hỗ trợ.
4. PDF native text + BM25/BGE-M3 cho born-digital subset, để tách lỗi OCR khỏi lỗi retriever.

**Visual retrieval**

5. SigLIP/Jina-CLIP multilingual single-vector.
6. ColPali zero-shot.
7. ColQwen/phi/modern ColVision checkpoint mạnh nhất tương thích tại ngày khóa benchmark.
8. Model Vietnamese-adapted đề xuất.

API proprietary như `text-embedding-3-large` là optional vì reproducibility và chi phí; không để nó là baseline bắt buộc.

### 7.2. Metrics

- **Primary:** macro nDCG@5 theo domain.
- **Secondary:** nDCG@1/10, Recall@1/5/10, MRR@10.
- **Graded relevance:** nDCG dùng nhãn 0/1/2; Recall báo cả fully relevant và any relevant.
- **Efficiency:** pages/s indexing, query latency P50/P95, QPS, index GB, RAM/VRAM.
- **Subgroups:** domain, page type, query type, human vs synthetic, born-digital vs scan và quality quartile.

Báo bootstrap 95% CI và paired randomization/bootstrap test trên query; hiệu chỉnh multiple comparisons khi so nhiều model.

### 7.3. Protocol công bằng

- Cùng corpus và qrels; không đánh giá mỗi model trên OCR output khác nhau mà không báo rõ.
- Tune hyperparameter trên dev; test chỉ chạy sau khi khóa pipeline.
- Báo exact model ID/revision, prompt, resolution, dtype, quantization, hardware và batch size.
- Page-level score từ text chunks phải dùng rule cố định, ví dụ max chunk score; ablate nếu kết luận phụ thuộc rule.
- Chạy ViDoRe gốc để kiểm tra catastrophic forgetting sau Vietnamese adaptation.

---

## 8. Error analysis cần có

Lấy mẫu stratified từ bốn nhóm: cả hai đúng, visual đúng/OCR sai, OCR đúng/visual sai, cả hai sai. Gán nguyên nhân:

- OCR diacritics/number/reading-order error;
- tiny text hoặc low resolution;
- table row/column confusion;
- chart legend/color;
- legal cross-reference;
- lexical shortcut hoặc duplicate template;
- query ambiguity/multiple relevant pages;
- annotation error.

Kết luận “OCR-free vượt OCR” chỉ hợp lệ nếu phân tích tách born-digital native text, OCR tốt và OCR lỗi; tránh dựng baseline OCR yếu làm đối chứng giả.

---

## 9. Kế hoạch 16 tuần và go/no-go

| Tuần | Công việc | Deliverable/Go criterion |
| :--- | :--- | :--- |
| 1--2 | Systematic literature scan; audit license; pilot 500 trang. | Xác nhận khoảng trống và ít nhất 3 nguồn có thể phát hành. |
| 3--4 | Pipeline render, provenance, dedup, PII và schema. | Near-duplicate audit chạy được; dataset card skeleton. |
| 5--7 | Viết/sinh query, relevance pooling, annotation pilot. | Agreement $\ge0,65$ hoặc sửa guideline và pilot lại. |
| 8 | Khóa dev/test v1 và contamination report. | Không có PDF/template leakage đã biết. |
| 9--10 | Chạy OCR/text và zero-shot visual baselines. | Reproducible harness; xác định error slices. |
| 11--12 | LoRA adaptation + hard-negative tuning một backbone. | Cải thiện dev trên ít nhất 3/5 subgroup, không chỉ average. |
| 13--14 | Test một lần, CI/significance, efficiency và error analysis. | Có bảng main results và limitations trung thực. |
| 15--16 | Dataset/model card, code, artifact và paper. | Có script dựng/evaluate lại và takedown policy. |

**Go/no-go:**

- Nếu không đủ license để phát hành ảnh: phát hành URL/hash/reconstruction scripts hoặc đổi nguồn.
- Nếu agreement thấp: giảm taxonomy, viết lại guideline; không mở rộng annotation trước khi pilot đạt chuẩn.
- Nếu fine-tuning không vượt zero-shot: bài vẫn có thể là benchmark/resource paper với negative result và error analysis; không ép claim model mới.
- Nếu nhân lực ít hơn 3 annotator part-time: giảm test xuống khoảng 1.500 query nhưng tăng chất lượng double annotation.

---

## 10. Bảng kết quả dự kiến cần điền

| Model | Input | Legal | Finance | Health | Edu/Info | Macro nDCG@5 | P95 ms | Index GB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native text + BM25 | Text | | | | | | | |
| OCR + BGE-M3 | Text | | | | | | | |
| SigLIP baseline | Image | | | | | | | |
| ColPali zero-shot | Image | | | | | | | |
| ColQwen zero-shot | Image | | | | | | | |
| Vietnamese-adapted | Image | | | | | | | |

Mọi con số như “vượt 15--25% nDCG@5” là mục tiêu thăm dò, không được đặt trong abstract/paper outline như kết quả đã biết.

---

## 11. Cấu trúc bài báo đề xuất

1. **Introduction:** nhu cầu retrieval trang tài liệu Việt và ba contribution có thể kiểm chứng.
2. **Related Work:** multilingual IR, visual document retrieval, Vietnamese document understanding và benchmark construction.
3. **Vi-ViDoRe:** collection, query taxonomy, judgments, split, ethics, statistics và contamination.
4. **Vietnamese Adaptation:** backbone, training data, negatives và objective.
5. **Experiments:** baseline, protocol, main results, efficiency và cross-benchmark retention.
6. **Analysis:** subgroup, human-vs-synthetic, OCR-vs-visual và qualitative errors.
7. **Limitations, Ethics and Broader Impact:** license, demographic/domain bias, misuse và takedown.
8. **Conclusion.**

---

## 12. Tài liệu nền tối thiểu

- Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*.
- ViDoRe benchmark, dataset cards và leaderboard hiện hành.
- ColQwen/ColQwen2.5 technical report và model card của checkpoint sử dụng.
- Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity*.
- Tài liệu gốc của PaddleOCR/VietOCR và Vietnamese text retriever được chọn.
- Datasheets for Datasets, Data Statements for NLP và hướng dẫn ACL về ethics/reproducibility.

Trước data freeze cần cập nhật systematic review tới ngày cutoff, tìm cả tiếng Anh lẫn tiếng Việt và lưu search strings, databases, inclusion criteria cùng danh sách excluded works.
