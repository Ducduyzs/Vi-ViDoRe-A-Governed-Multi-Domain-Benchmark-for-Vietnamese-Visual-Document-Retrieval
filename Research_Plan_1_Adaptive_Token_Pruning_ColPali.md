# Đề cương Nghiên cứu 1: Adaptive Token Budgeting for Visual Late-Interaction Retrieval

> **Tên dự kiến:** *AdaBudget-Col: Content-Adaptive Token Budgeting for Efficient Visual Late-Interaction Retrieval*  
> **Mục tiêu phù hợp:** CIKM/SIGIR short paper, ECIR, LIR workshop; main track chỉ khi có đánh giá hệ thống quy mô lớn.  
> **Thời gian:** 12--16 tuần.

---

## 1. Tóm tắt và đánh giá

ColPali/ColQwen biểu diễn mỗi trang bằng hàng trăm đến hàng nghìn vector và chấm điểm bằng MaxSim. Với 1 triệu trang, 1024 token/trang, 128 chiều và FP16, riêng embedding thô chiếm:

$$10^6\times1024\times128\times2\ \text{bytes}\approx262\ \text{GB}.$$

Hướng nghiên cứu có giá trị thực tiễn cao, nhưng novelty ban đầu bị chồng lấn với token pooling, ToMe, PLAID, DESSERT và MUVERA. Định vị có thể bảo vệ được là: **phân bổ ngân sách token theo độ phức tạp từng trang, bảo toàn vùng hiếm/quan trọng và tối ưu trực tiếp quality--cost frontier**.

| Tiêu chí | Điểm | Nhận xét |
| :--- | :---: | :--- |
| Tính mới | 7/10 | Khá nếu vượt pooling tĩnh ở cùng byte/latency; chỉ ghép entropy + merging + MRL chưa đủ mới. |
| Tác động | 8,5/10 | Giải quyết trực tiếp index size và MaxSim cost. |
| Khả thi | 7/10 | Zero-shot/scorer-only khả thi; thêm MRL và engine mới cùng lúc là quá rộng. |
| Rủi ro | Cao | Attention không đồng nghĩa salience; FLOPs giảm chưa chắc latency giảm. |

**Khuyến nghị:** chọn adaptive token budgeting làm đóng góp chính. MRL và early exit là phần mở rộng tùy kết quả, không phải điều kiện hoàn thành bài.

---

## 2. Bối cảnh và khoảng trống

### 2.1. Vấn đề

- **Storage:** multi-vector index lớn hơn single-vector nhiều bậc.
- **MaxSim cost:** với $N_q$ query token và $N_d$ document token, exact scoring có chi phí $O(N_qN_dd)$.
- **Dư thừa không đồng đều:** trang thưa có nhiều token gần nhau; bảng nhỏ và trang dày chữ cần ngân sách lớn hơn.
- **Chi phí hệ thống:** pruning/merging chỉ hữu ích nếu overhead nhỏ hơn phần tính toán tiết kiệm được.

### 2.2. Related work và novelty hợp lệ

- ColPali đã báo cáo hierarchical token pooling hệ số 3 giảm 66,7% vector và giữ 97,8% hiệu năng trung bình; trang dày chữ suy giảm nhiều hơn.
- PLAID/DESSERT tối ưu candidate generation và scoring cho late interaction.
- MUVERA xấp xỉ multi-vector similarity bằng fixed-dimensional encoding để lấy ứng viên, sau đó vẫn có thể rerank bằng MaxSim gốc. Không nên mô tả MUVERA đơn giản là làm mất late interaction.
- ToMe và token pruning trong ViT chủ yếu tối ưu forward pass; mục tiêu ở đây là **representation lưu trữ và retrieval quality sau khi document đã encode**.

Khoảng trống: chưa đủ bằng chứng về một bộ điều khiển **query-independent tại inference, nhưng học từ query distribution của train set**, phân bổ $K_d$ theo trang và thắng static pooling trên Pareto frontier qua nhiều loại trang/backbone.

---

## 3. Câu hỏi nghiên cứu và giả thuyết

- **RQ1:** Adaptive budgeting có tốt hơn pooling tĩnh ở cùng số byte/trang không?  
	**H1:** tại mức giảm ít nhất $4\times$ số vector, nDCG@5 giảm không quá 2% tương đối và cao hơn static pooling cùng budget.
- **RQ2:** Thành phần nào tạo lợi ích: novelty, spatial constraint, learned retrieval prior hay merge?  
	**H2:** learned prior giúp text/table-heavy; spatial merge giúp background/chart-heavy.
- **RQ3:** Phương pháp có tổng quát sang domain/backbone chưa thấy không?  
	**H3:** Pareto hypervolume tăng trên ít nhất một backbone hoặc dataset ngoài miền huấn luyện.
- **RQ4:** Giảm phép tính có chuyển thành tăng tốc thật không?  
	**H4:** P95 reranking nhanh hơn ít nhất $2\times$ tại quality target đã chọn, tính cả overhead.

Các ngưỡng trên là **target falsifiable**, không phải kết quả được tuyên bố trước.

---

## 4. Phương pháp đề xuất

### 4.1. Salience không phụ thuộc query test

Với token $d_i$, dùng:

$$S_i=\alpha s_i^{\text{novelty}}+\beta s_i^{\text{attn}}+\gamma s_i^{\text{layout}}+\delta s_i^{\text{retrieval}},$$

trong đó:

- $s_i^{\text{novelty}}=1-\max_{j\in\mathcal{N}(i)}\cos(d_i,d_j)$ đo độ mới so với vùng lân cận.
- $s_i^{\text{attn}}$ là attention rollout hoặc attribution đã chuẩn hóa theo layer/head; không giả định entropy cao luôn là nội dung quan trọng.
- $s_i^{\text{layout}}$ là prior từ edge density hoặc layout/OCR box, chỉ dùng trong ablation hybrid để giữ nhánh chính OCR-free.
- $s_i^{\text{retrieval}}$ ước lượng xác suất token trở thành argmax của query token, chỉ học từ training queries.

Document index phải độc lập với query runtime. Vì vậy thuật ngữ đúng là **query-distribution-aware prior**, không phải query-aware indexing.

### 4.2. Dynamic budget controller

Tính độ phức tạp trang $c(d)$ từ dispersion embedding, salience histogram, spatial occupancy và token redundancy. Bộ điều khiển chọn:

$$K_d\in\{64,128,256,512\},\qquad \mathbb{E}_{d}[K_d]\le B.$$

So sánh ba controller: rule-based quantile, learned regressor và differentiable budget predictor.

### 4.3. Spatial semantic merge-then-prune

Tạo đồ thị lân cận không gian, chỉ merge token gần nhau có cosine vượt $\tau$ và không thuộc nhóm protected high-salience. Vector cụm được chuẩn hóa lại:

$$d_{C_k}=\frac{\sum_{j\in C_k}w_jd_j}{\left\|\sum_{j\in C_k}w_jd_j\right\|_2}.$$

Sau merge, giữ top-$K_d$ token theo salience. Normalization là bắt buộc vì MaxSim giả định hướng vector có ý nghĩa; mean không chuẩn hóa có thể làm sai scale score.

### 4.4. Training objective

Teacher là full-token retriever. Student tối ưu:

$$\mathcal{L}=\mathcal{L}_{\text{rank}}+\lambda_1\mathcal{L}_{\text{score-distill}}+\lambda_2\mathcal{L}_{\text{rank-distill}}+\lambda_3\max(0,\mathbb{E}[K_d]-B).$$

Ba mức thực nghiệm: train-free, scorer/controller-only và LoRA end-to-end. Chỉ thêm Matryoshka dimensions $\{32,64,128\}$ nếu adaptive token core đã thắng baseline.

---

## 5. Thiết kế thực nghiệm

### 5.1. Dữ liệu

- ViDoRe: báo riêng từng subtask và macro-average.
- Một tập cross-domain không dùng khi fit controller; ưu tiên benchmark visual-document có license rõ ràng.
- Một corpus lớn để đo hệ thống; nếu dùng trang không nhãn, chỉ đo index/latency, không trộn vào quality test.

Chia train/dev/test theo **PDF/document**, không theo page. Threshold, budget và stopping rule chỉ chọn trên dev.

### 5.2. Baseline

1. Full-token exact MaxSim oracle.
2. Random dropping, uniform grid, top-$K$ magnitude.
3. Hierarchical mean token pooling và k-means/centroid pooling.
4. Fixed-$K$ salience pruning để tách lợi ích scorer khỏi dynamic budget.
5. PLAID/PyLate hoặc MUVERA nếu tương thích; báo riêng search/index optimization với representation compression.
6. Ít nhất ColPali và một checkpoint ColQwen/ColQwen2.5 được khóa revision.

Không mặc định mọi backbone có 1024 token. Phải báo token count thực tế theo resolution và processor.

### 5.3. Metrics

- **Quality:** nDCG@5 (chính), nDCG@10, Recall@1/5/20, MRR@10.
- **Compression:** vectors/page, bytes/page, total index GB, compression ratio.
- **System:** indexing pages/s, scorer ms/page, mean/P50/P95/P99 latency, QPS, RAM/VRAM và GPU-hours.
- **Robustness:** worst-group quality drop trên text-, table-, chart-heavy, scan nhiễu và trang nhiều khoảng trắng.

Storage phải gồm vector, codebook/centroid, metadata và cấu trúc index.

### 5.4. Protocol thống kê và công bằng

- So sánh ở ba chế độ: cùng số vector, cùng số byte và cùng latency.
- Chạy exact MaxSim trước để tách lỗi representation khỏi lỗi ANN.
- Bootstrap confidence interval theo query; tối thiểu 3 seed cho thành phần học được.
- Khóa model revision, commit, resolution, dtype, batch size và hardware.
- Không dùng cùng PDF/source/template ở train và test; kiểm tra exact hash và near-duplicate bằng perceptual hash.

### 5.5. Ablation bắt buộc

- Fixed $K$ vs dynamic $K_d$.
- Prune-only vs merge-only vs merge-then-prune.
- Bỏ từng thành phần novelty, attention, spatial và retrieval prior.
- Mean merge vs normalized salience-weighted merge.
- Sweep $K$, $\tau$ và budget $B$.
- FP16/BF16 vs INT8/PQ; 128d vs 64d/32d chỉ khi có MRL training.
- In-domain vs cross-domain; ColPali vs Qwen backbone.

Kết quả chính phải là ba Pareto plot: nDCG@5--bytes/page, nDCG@5--P95 latency và recall--QPS; bổ sung Pareto hypervolume.

---

## 6. Kế hoạch 12 tuần và tiêu chí go/no-go

| Tuần | Công việc | Tiêu chí |
| :--- | :--- | :--- |
| 1--2 | Reproduce exact MaxSim và hierarchical pooling; khóa harness. | Sai khác không quá 1 điểm nDCG@5 so với cấu hình tham chiếu. |
| 3--4 | Phân tích redundancy, argmax-hit và oracle budget theo page type. | Có bằng chứng static budget gây under/over-allocation. |
| 5--6 | Adaptive budget + spatial merge train-free. | Không bị static pooling thống trị trên toàn Pareto frontier. |
| 7--8 | Train controller/distillation. | Ít nhất $4\times$ compression, giảm $\le2\%$ nDCG@5 tương đối average và $\le5\%$ worst group. |
| 9--10 | Cross-backbone, ablation và system benchmark. | Có speedup P95; nếu không, chỉ claim storage compression. |
| 11--12 | Error analysis, artifact và paper. | Bảng reproducibility và scripts chạy lại đầy đủ. |

**Fallback:** nếu adaptive method không thắng static pooling, chuyển đóng góp thành nghiên cứu hệ thống về compressibility theo page type và oracle allocation; bỏ MRL/early exit. Nếu quality tốt nhưng không tăng tốc wall-clock, không tuyên bố acceleration.

---

## 7. Bảng kết quả dự kiến cần điền

| Method | Vectors/page | Bytes/page | nDCG@5 | Worst-group drop | P95 ms | QPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full tokens | | | | | | |
| Static pooling | | | | | | |
| Fixed-$K$ salience | | | | | | |
| AdaBudget-Col train-free | | | | | | |
| AdaBudget-Col trained | | | | | | |

Không ghi trước các mức tăng 15--25% hoặc giữ trên 98% vào abstract/kết luận trước khi có số liệu.

---

## 8. Tài liệu nền tối thiểu

- Faysse et al., *ColPali: Efficient Document Retrieval with Vision Language Models*.
- Clavie et al., *Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling*.
- Bolya et al., *Token Merging: Your ViT But Faster*.
- Santhanam et al., *PLAID*; Engels et al., *DESSERT*.
- Dhulipala et al., *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings*.
- Kusupati et al., *Matryoshka Representation Learning*.

Trước khi nộp bài cần chạy lại systematic literature search theo ngày cutoff và ghi rõ phiên bản arXiv/model checkpoint vì lĩnh vực này thay đổi nhanh.
