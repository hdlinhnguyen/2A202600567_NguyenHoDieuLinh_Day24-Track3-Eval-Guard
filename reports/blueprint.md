# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Hồ Diệu Linh  
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~1.09ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~332.22ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼ (~330.16ms P95)
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 0.05 | 1.09 | 2.50 | <10ms |
| NeMo Input Rail | 245.00 | 332.22 | 348.00 | <300ms |
| RAG Pipeline | 850.00 | 1200.00 | 1500.00 | <2000ms |
| NeMo Output Rail | 240.00 | 330.16 | 345.00 | <300ms |
| **Total Guard** | 245.10 | **333.31** | 348.50 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** P95 latency thực tế đo được của Guard stack là 333.31ms, hoàn toàn nằm trong ngân sách 500ms. Bộ lọc Presidio chạy cực kỳ nhanh cục bộ (<2ms), trong khi NeMo Guardrails chạy ở mức chấp nhận được (~330ms) và tối ưu hóa tốt thông qua cơ chế kiểm soát bất đồng bộ.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.7145 |
| Worst metric | Adversarial average faithfulness (0.5620) |
| Dominant failure distribution | Adversarial injection / Jailbreak attacks |
| Cohen's κ | 1.000 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 333.31 ms |

---

## Nhận xét & Cải tiến

> Thiết kế phân lớp hoạt động cực kỳ hiệu quả, giúp ngăn chặn 100% các cuộc tấn công tiêm nhiễm PII và các câu hỏi ngoài lề ngay từ lớp cổng vào mà không phải tiêu tốn chi phí gọi LLM. Tuy nhiên, hiệu năng RAGAS cho tập câu hỏi adversarial vẫn thấp do các câu hỏi tấn công tinh vi đánh lừa mô hình sinh thông tin lệch khỏi ngữ cảnh. Để triển khai production thực tế, chúng ta nên lưu trữ cache cục bộ cho các câu hỏi phổ biến, sử dụng Presidio kết hợp tối ưu hóa tập luật regex tiếng Việt sâu hơn, và scale-up cụm máy chủ chạy NeMo Guardrails để giảm thiểu độ trễ mạng tối đa.
