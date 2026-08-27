# Human actions required before Vi-ViDoRe freeze

The automated work has produced a governed candidate, not final qrels. Complete these items in order.

## A. Data owner

- Verify the original source URL and redistribution terms for every registry row marked `needs_source_review`.
- Replace or remove any test document whose rights cannot be verified.
- Acquire at least one additional independent source for legal, financial, and healthcare so domain is not identical to publisher.
- Acquire Vietnamese form/template pages with explicit redistribution permission.
- Mark whether each legacy PDF is complete; rebuild documents currently recorded as `first_10` from the full PDF.
- Approve `ANNOTATION_GUIDELINE_V1.md` and record the approval date/version.

## B. Annotation lead

- Write or edit at least 500 Vietnamese test queries, with at least 50 per required domain.
- Ensure at least 40% are genuinely human-written.
- Include at least 20 judged target queries for scans and for each required page type.
- Pool candidate pages from BM25, a Vietnamese dense retriever, and a visual retriever.
- Obtain two independent judgments for every pooled query-page pair.
- Adjudicate disagreements and calculate agreement before adjudication.
- Save final rows as `data/benchmark_governed_v0_1/test/annotations_final.tsv`.

## C. ML/evaluation owner

- Do not tune on the candidate or final test set.
- Generate queries for the two new World Bank documents only through the approved human workflow.
- Preserve per-query rankings for every baseline used in pooling.
- Re-run `python -m scripts.build_governed_benchmark` after registry or annotations change.
- Run `python -m scripts.build_governed_benchmark --freeze` only when the report shows all gates passing.

## D. Current non-negotiable blockers

- Six test PDFs have unverified redistribution rights.
- Six legacy test PDFs contain only the first ten processed pages.
- No final human annotations exist.
- Education has 11 test queries and legal has 15; each required domain needs at least 50.
- No test query currently targets a scanned page.
- No final query targets a chart or form/template page.
- The candidate has no finance query for the newly added finance document.

