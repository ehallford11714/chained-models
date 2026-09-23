# Chained Models

Retrieve a hierarchical PageIndex, then a trained router picks the next frozen specialist. The second hop sees the first hop’s note plus a fresh page fetch. This is an assembly of published ideas (MoE routing, PageIndex, Socratic handoff). The ablation below is the claim.

## Experiment

Six conditions on the same planted corpus (code, SQL, science, law, finance). Each document has three headings; only one heading holds the gold fact. Flat RAG sees undifferentiated chunks. PageIndex ranks the document tree first, then pages under the winning heading.

| Condition | Retrieval | Hops |
| --- | --- | --- |
| `direct` | none | 1 |
| `flat` | bag-of-words over all chunks | 1 |
| `pageindex` | heading tree, then pages | 1 |
| `chain` | none | 2 |
| `chain_flat` | flat at each hop | 2 |
| `chain_pageindex` | PageIndex at each hop | 2 |

Metrics: page recall@3, gold-heading hit, route accuracy, extractive answer match. Specialists are closed-book stubs: they only emit the fact if their domain matches **and** the retrieved pages contain it. That isolates routing and retrieval. It is not a 7B generation benchmark.

Measured on 8 cases (5 single-hop, 3 two-hop):

| Condition | Recall | Heading | Route | Answer |
| --- | --- | --- | --- | --- |
| `direct` | 0.00 | 0.00 | 0.625 | 0.00 |
| `flat` | 1.00 | 0.875 | 0.625 | 0.625 |
| `pageindex` | 1.00 | 0.875 | 0.625 | 0.625 |
| `chain` | 0.00 | 0.00 | 0.625 | 0.00 |
| `chain_flat` | 1.00 | 0.875 | 0.875 | 0.875 |
| `chain_pageindex` | 1.00 | 0.875 | 0.875 | **0.875** |

Chaining without retrieval cannot answer. Retrieval without a second hop misses the composed cases. PageIndex and flat matched on this corpus (gold words sit in the queries). The product default is `chain_pageindex`: same score as `chain_flat`, heading tree ready for longer docs.

## Suite

```
python -m pagechain ablate
python -m pagechain train
python -m pagechain index examples/docs
python -m pagechain ask What PAM does the lab note use?
```

`train` writes `checkpoints/entry.pt` and `checkpoints/hop.pt`. `index` rebuilds the PageIndex from markdown headings. `ask` runs the default product path: PageIndex, then up to two trained hops. `ChainedModels` is the same API in Python.

