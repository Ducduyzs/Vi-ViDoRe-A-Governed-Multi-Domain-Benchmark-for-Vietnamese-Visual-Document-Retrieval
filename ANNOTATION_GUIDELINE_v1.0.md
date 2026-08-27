# Vi-ViDoRe Annotation Guideline v1.0

**Version**: 1.0  
**Date**: 2026-08-27  
**Benchmark**: Vietnamese Visual Document Retrieval (Vi-ViDoRe)

---

## 1. Mục tiêu
Gán nhãn độ liên quan (relevance) giữa **câu hỏi (query)** và **trang tài liệu (page)** để xây dựng qrels (ground truth) cho benchmark Vi-ViDoRe.

---

## 2. Thang điểm Relevance (0 / 1 / 2)

| Điểm | Nhãn | Định nghĩa | Ví dụ |
|------|------|------------|-------|
| **0** | Not Relevant | Trang **không chứa** thông tin trả lời câu hỏi. Có thể cùng domain nhưng nội dung không liên quan. | Query: "Mức phạt vi phạm an toàn lao động" → Trang nói về "Quy trình tuyển dụng" |
| **1** | Partially Relevant | Trang chứa **một phần** thông tin trả lời, hoặc chứa ngữ cảnh/hỗ trợ nhưng **không đủ** để trả lời hoàn chỉnh. | Query: "Quy trình xin visa LĐ Việt Nam sang Hàn Quốc" → Trang chỉ liệt kê "Hồ sơ cần chuẩn bị" mà không có quy trình nộp/hồ sơ tiếp nhận |
| **2** | Fully Relevant | Trang chứa **đầy đủ** thông tin để trả lời câu hỏi một cách độc lập (stand-alone). Bao gồm: điều khoản pháp lý, số liệu bảng, định nghĩa, quy trình, phác đồ, câu trả lời trực tiếp. | Query: "Mức phạt vi phạm an toàn lao động theo Luật 2019" → Trang chứa chính xác Điều 45, Khoản 2, Mức phạt 20-50 triệu |

---

## 3. Nguyên tắc cốt lõi

### 3.1 Đánh giá trên nội dung TRANG (page-level)
- Chỉ nhìn **nội dung trang được gán** (native text + hình ảnh). Không dùng kiến thức ngoài, không đoán mò trang khác.
- Nếu cần lật trang trước/sau để hiểu → **không phải fully relevant (2)**.

### 3.2 Query tự đủ (stand-alone)
- Query đã được làm sạch: **không có từ chỉ vị trí** ("trang này", "hình trên", "bảng dưới", "theo văn bản").
- Nếu query vẫn mơ hồ → gán **0** hoặc **1** tùy mức độ mơ hồ.

### 3.3 Loại bằng chứng (Evidence types)
| Loại | Cách gán điểm 2 |
|------|-----------------|
| **Văn bản pháp lý** | Có đầy đủ Điều/Khoản/Điểm, mức phạt, phạm vi áp dụng |
| **Số liệu bảng/biểu mẫu** | Có đúng chỉ tiêu, đơn vị, kỳ kế toán, giá trị số |
| **Định nghĩa/khái niệm** | Có định nghĩa rõ ràng, có ví dụ/minh họa nếu cần |
| **Quy trình/thủ tục** | Có các bước tuần tự, cơ quan nhận, thời hạn, hồ sơ |
| **Biểu đồ/infographic** | Có nhãn trục, đơn vị, năm, so sánh được yêu cầu |
| **Scanned/OCE** | Đọc được chữ, không bị mờ/mất chữ quan trọng |

---

## 4. Các trường hợp đặc biệt

| Tình huống | Hướng dẫn |
|------------|-----------|
| **Nhiều trang liên quan** | Mỗi trang gán độc lập. Có thể có nhiều trang **2** cho 1 query. |
| **Trang mục lục/chỉ mục** | Gán **1** nếu trỏ đến trang có nội dung; **0** nếu chỉ liệt kê tiêu đề. |
| **Trang bìa/phụ lục** | Gán **0** trừ khi chứa chính xác thông tin query cần. |
| **Bảng số liệu không đầy đủ** | Thiếu đơn vị/thời gian/chỉ tiêu → **1** |
| **Hình ảnh/scan không đọc được** | Gán **0** (không thể verify). Ghi chú `evidence_note: "unreadable_scan"` |
| **Query mơ hồ/đầu vào kém** | Gán **0**, ghi `evidence_note: "ambiguous_query"` |

---

## 5. Quy trình Double Annotation + Adjudication

1. **Annotator A** và **Annotator B** gán nhãn độc lập (không xem nhãn của nhau).
2. Tính **Cohen's κ** trên cặp (query, page).
   - κ ≥ 0.67 → chấp nhận, lấy **majority vote** (nếu 2 annotator → lấy điểm cao hơn để an toàn recall).
   - κ < 0.67 → **Adjudicator** (người thứ 3) xem lại guideline, thảo luận, đưa ra nhãn cuối.
3. Lưu cả 3 nhãn: `annotator_a`, `annotator_b`, `adjudicated_relevance`.

---

## 6. Định dạng file Annotation (TSV)

| Cột | Mô tả | Ví dụ |
|-----|-------|-------|
| `query_id` | ID câu hỏi | `q_test_05_mang_may_tinh_p01_01` |
| `page_id` | ID trang | `05_mang_may_tinh_p01` |
| `annotator_id` | Mã annotator | `annotator_A` |
| `relevance` | 0 / 1 / 2 | `2` |
| `query_status` | `PENDING` / `JUDGED` / `ADJUDICATED` | `JUDGED` |
| `evidence_note` | Ghi chú bằng chứng | `full_answer_in_table_3` |
| `judged_at` | Timestamp ISO | `2026-08-27T10:30:00` |
| `guideline_version` | Phiên bản guideline | `1.0` |
| `adjudicated_relevance` | Nhãn sau adjudication (nếu có) | `2` |
| `candidate_source_page` | Trang này có phải candidate gốc không | `true` / `false` |

---

## 7. Checklist trước khi submit

- [ ] Đã đọc toàn bộ Guideline v1.0
- [ ] Đã làm pilot 10 queries cùng annotator kia → tính κ
- [ ] κ ≥ 0.67? Nếu không → discuss + update guideline
- [ ] Mỗi query-page pair có 2 nhãn độc lập
- [ ] Không bỏ sót bất kỳ pair nào trong pool
- [ ] `evidence_note` mô tả vị trí bằng chứng (điều khoản, ô bảng, đoạn văn)

---

## 8. Liên hệ / Cập nhật

- Guideline owner: [Tên/Email]
- Issue tracker: GitHub Issues (tag `annotation`)
- Version control: Mọi thay đổi guideline phải bump version (1.1, 1.2...) và ghi changelog.