# Vi-ViDoRe relevance annotation guideline

Version: 1.0  
Unit of judgment: one query-page pair

## 1. Annotator task

Read the query without assuming knowledge of its source page. Inspect the candidate page image and decide whether the page supplies the information needed by the query.

Do not reward a page merely because it shares keywords. Judge answerability and evidence.

## 2. Labels

- **2 - Fully relevant:** The page directly contains enough evidence to answer the query or complete the requested comparison.
- **1 - Partially relevant:** The page contains useful evidence but lacks an essential value, condition, row/column, legend, clause, or continuation needed for a complete answer.
- **0 - Not relevant:** The page does not answer the query, contains only superficial lexical overlap, or refers elsewhere without useful evidence.

Use status `UNANSWERABLE` when no pooled page answers the query. Use `AMBIGUOUS` when multiple reasonable interpretations would change the relevance label. Use `DUPLICATE_QUERY` for semantic query duplicates.

## 3. Page-type guidance

- **Tables:** verify the correct row, column, unit, period, and entity.
- **Charts:** verify the legend, axis, category, time range, and visual comparison.
- **Legal/policy:** verify that the requested rule, condition, scope, or authority is present; general discussion is not a substitute for a rule.
- **Health:** do not infer medical recommendations that the page does not state.
- **Scans:** judge visible evidence, not OCR quality. Flag unreadable pages separately.
- **Multi-page answers:** label each page independently. A page carrying only half of the evidence is normally label 1.

## 4. Query validation before relevance labeling

Reject or repair a query when it:

- refers to “this page/table/figure”;
- cannot be understood without the source page;
- contains a hallucinated fact or asks for information absent from the document;
- copies a uniquely identifying sentence or full title;
- is not Vietnamese for the Vietnamese test set;
- has the same meaning as an existing query;
- asks for unsafe personal, medical, or financial information.

Record whether the final query is `human_written`, `llm_edited`, or `llm_accepted`. “LLM-assisted” alone is insufficient for the final dataset card.

## 5. Annotation workflow

1. Validate the query.
2. Judge the known source page and pooled top results from all baseline families.
3. Search within the same PDF for additional relevant pages.
4. Submit an independent label without seeing the other annotator.
5. Send disagreements to an adjudicator.
6. Preserve the pre-adjudication labels for agreement calculation.

## 6. Required fields

Each annotation row must include:

- `query_id`, `page_id`, `annotator_id`;
- `relevance` in `{0,1,2}`;
- `query_status`;
- `evidence_note` describing visible evidence;
- `judged_at` timestamp;
- `guideline_version`;
- `adjudicated_relevance` when applicable.

## 7. Pilot acceptance

Run a 100-query pilot before full annotation. Discuss disagreements, revise examples, then repeat the pilot if Krippendorff's alpha or weighted Cohen's kappa is below 0.67. Report both pre- and post-adjudication statistics.

