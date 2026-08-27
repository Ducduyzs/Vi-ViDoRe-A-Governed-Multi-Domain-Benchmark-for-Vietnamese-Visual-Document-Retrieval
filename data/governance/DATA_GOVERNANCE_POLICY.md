# Vi-ViDoRe data governance policy

Version: 1.0  
Status: mandatory for every benchmark release

## 1. Release units

The project separates four artifacts:

1. `raw_pdfs`: immutable source files. A raw file is never silently edited or replaced.
2. `curated corpus`: pages that pass language, duplicate, provenance, and safety checks.
3. `candidate benchmark`: fixed document assignments plus queries awaiting human judgment.
4. `frozen benchmark`: a release that passes every automated and human gate.

Results intended for a paper may only use a frozen benchmark manifest. A candidate split is for annotation and development only.

## 2. Inclusion rules

A document may enter the Vietnamese benchmark only when:

- its primary document language is Vietnamese;
- its domain and source are manually reviewed;
- the source URL, publisher, access date, checksum, and license status are recorded;
- redistribution is allowed, or the release contains only URL, checksum, and reconstruction instructions;
- it contains no prohibited personal or sensitive information;
- it is not an exact or near duplicate of another included document;
- the complete-document or sampled-page status is explicit.

English documents are excluded from the Vietnamese benchmark. They may be released as a separately named cross-lingual evaluation set.

## 3. Leakage policy

The following units may never cross train/dev/test:

- exact PDF;
- document version family;
- publisher/source group when source-held-out evaluation is claimed;
- template cluster;
- translated or paraphrased copies of a query.

The registry is authoritative. Automated pHash/text checks supplement the registry but do not replace manual source and template review.

## 4. Relevance judgments

Automatically generated target pages are candidate labels, not qrels. Frozen qrels require:

- pooling from at least BM25, one dense Vietnamese retriever, and one visual retriever;
- labels `0 = not relevant`, `1 = partially relevant`, `2 = fully relevant`;
- two independent judgments for every test candidate pair;
- adjudication of every disagreement;
- explicit marking of ambiguous, unanswerable, or duplicate queries;
- reporting agreement before adjudication.

## 5. Freeze policy

The test set is frozen only after:

- all release-blocking license fields are verified;
- all required domains and page types meet the declared minimums;
- scan coverage is present among judged target pages;
- duplicate and group-leakage checks pass;
- final human annotations exist;
- query provenance and human/LLM origin are recorded;
- a manifest containing hashes of the registry, corpus, queries, and qrels is generated.

After freeze, test queries, qrels, pages, and split assignments are immutable. Corrections require a new benchmark version and a changelog.

## 6. Current release rule

The legacy benchmark is a pilot and must not be described as human-validated. The governed builder writes `FREEZE_BLOCKED.md` until all release gates pass.

