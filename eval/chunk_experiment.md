# DocChat-AI Chunking & Top-K Experiment

Exploration of text chunk sizes (`chunk_size`) and retrieval depth (`top_k`) across evaluation queries.

## Methodology & Grid

The grid search evaluates configurations across:
- **Chunk Sizes:** 400, 600, 800, 1000 characters
- **Chunk Overlap:** 12% of chunk size
- **Top-K:** 3, 4, 6 retrieved passages

Retrieval performance is evaluated against answerable questions using:
- **Chunk Hit@k (Headline):** Whether any retrieved chunk contains the expected answer text.
- **Page Hit@k:** Whether the target page was retrieved within the top-k chunks.
- **MRR (Mean Reciprocal Rank):** Reciprocal rank of the first relevant chunk.
- **Index Coverage:** Fraction of total document chunks retrieved per query (`k / chunks`).

## Key Observations

- **Index Coverage vs. Hit Rate:** Configurations with very large chunks yield few total chunks on shorter documents. At high `top_k`, this retrieves a large fraction of the entire document index, which inflates hit rate trivially while reducing answer specificity.
- **Information Granularity:** Smaller chunk sizes produce more focused passages and lower index coverage per query, ensuring retrieved chunks contain precise answers rather than multi-topic sections.
- **Ranking Quality:** Prioritizing MRR ensures relevant passages are ranked higher, placing the most pertinent context at rank 1 for generation.

## Running the Experiment

To run or re-evaluate the chunking experiment on your dataset:

```bash
python scripts/chunk_experiment.py --pdf <path_to_pdf> --questions <path_to_questions.json>
```