# Rousseau GraphRAG: Comparing Graph-Based and Vector-Based Retrieval

> **Template note:** Replace every `[TODO: ...]` with information from your actual project. The setup instructions and example questions below are guidance; confirm them against your files before publishing. Remove this note when finished.

## Overview

This project explores question answering over the writings of Jean-Jacques Rousseau using Microsoft's GraphRAG and a vector-based retrieval-augmented generation (VectorRAG) system.

The central question is: **How do these two retrieval approaches compare when answering questions about individual passages and themes that span multiple works?**

Rousseau's writings provide a useful setting for investigating connections between concepts such as freedom, equality, education, and political authority. This project examines whether graph-based retrieval improves the coverage and usefulness of answers, and what trade-offs appear in response time and processing cost.

**Project status:** [TODO: Describe what works now, what has been evaluated, and what remains unfinished.]

## My Contribution

This project applies Microsoft's GraphRAG framework to a selected Rousseau corpus. My work focuses on preparing the texts, configuring retrieval, implementing the comparison workflow, and interpreting the results.

[TODO: Keep only the contributions you have completed, and adjust their wording.]

- Collected and prepared the source texts for indexing.
- Configured GraphRAG's models, text chunking, and entity extraction prompts.
- Implemented a VectorRAG baseline using Python, LangChain, and Chroma.
- Developed a benchmark question set covering specific details and broader philosophical themes.
- Built scripts to generate answers, evaluate their quality, and analyze response times.
- Documented the strengths and limitations of each approach.

## How the Systems Work

| Stage | VectorRAG | GraphRAG |
| --- | --- | --- |
| Preparation | Splits source texts into passages and embeds them. | Splits texts, extracts entities and relationships, and builds a graph and community reports. |
| Retrieval | Finds passages similar to the question using vector search. | Uses the configured search method to retrieve graph-related context and/or community reports. |
| Answer generation | Supplies retrieved passages to a language model. | Supplies the selected context to a language model. |

GraphRAG supports different search methods. **The method evaluated here is [TODO: global, local, DRIFT, or another method actually used].** Describe its actual retrieval behavior rather than treating all GraphRAG searches as equivalent.

## Source Texts

[TODO: Add one row for every work actually indexed. Record the exact translation and edition because these affect wording and reproducibility.]

| Work | Translator / edition | Source URL | Used in GraphRAG? | Used in VectorRAG? |
| --- | --- | --- | --- | --- |
| [TODO: Work title] | [TODO: Translator and edition] | [TODO: Direct source link] | [TODO: Yes/no] | [TODO: Yes/no] |

**Text preparation:** [TODO: Explain removal of headers, footers, notes, or other material; file encoding; and whether books were separate files or combined.]

**Corpus size:** [TODO: Number of works and approximate word or token count, stating the unit.]

Full texts are [TODO: included / excluded] from this repository. If excluded, use the source links above and follow [TODO: preparation instructions] to reproduce the input files. Check redistribution permissions for the specific translations and editions before including them.

## Configuration

Record the settings used for the reported benchmark, including settings that differ between indexing and answering.

| Setting | VectorRAG | GraphRAG |
| --- | --- | --- |
| Framework and version | [TODO: LangChain and Chroma versions] | [TODO: Exact GraphRAG version] |
| Embedding model | [TODO: Exact model ID] | [TODO: Exact model ID] |
| Answer-generation model | [TODO: Exact model ID] | [TODO: Exact model ID] |
| Indexing / extraction model | Not applicable unless used | [TODO: Exact model ID] |
| Chunk size and unit | [TODO: Value; characters or tokens] | [TODO: Value; characters or tokens] |
| Chunk overlap and unit | [TODO: Value and unit] | [TODO: Value and unit] |
| Retrieval settings | [TODO: Search method and number of passages] | [TODO: Search method, community level, and context limits as applicable] |
| Answer instructions | [TODO: Prompt file and evidence rules] | [TODO: Prompt file and evidence rules] |

See `settings.yaml` and the relevant prompt files for the configuration. Keep credentials out of these files.

## Repository Contents

[TODO: Adjust this table to match the files you actually upload.]

| Path | Purpose |
| --- | --- |
| `benchmark/` | Benchmark questions, answer-generation scripts, evaluation scripts, and small results files. |
| `RAG application/` | VectorRAG implementation, including ingestion and retrieval code. |
| `src/` | Additional project source code. |
| `prompts/` | Prompts used by the GraphRAG workflow. |
| `settings.yaml` | GraphRAG configuration. |
| `requirements.txt` | Python dependencies and versions needed to reproduce the project. |
| `.env.example` | Required environment variable names with placeholder values. |
| `.gitignore` | Excludes credentials, environments, and generated artifacts. |

Generated databases, caches, logs, and full indexing outputs are excluded. They can be recreated locally. [TODO: If rebuilding is expensive, document the observed indexing time and cost, or link to an optional downloadable artifact.]

## Getting Started

> These are setup examples. Replace the repository placeholder and confirm the commands against your installed version and script locations before publishing.

### 1. Download the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create a Python environment

**Python version used:** [TODO: Exact tested Python version.]

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[TODO: Create and verify `requirements.txt` before publishing these instructions.]

### 3. Configure credentials

Copy `.env.example` to `.env`, then supply your own API key locally. Use the variable names referenced by `settings.yaml` and your Python code.

[TODO: List required variable names here, with placeholders only.]

Never commit `.env` or real API keys. Running indexing and queries with hosted models incurs API usage charges.

### 4. Prepare and index the texts

Place the prepared texts in `input/`, or the input directory specified in `settings.yaml`.

[TODO: Insert your tested GraphRAG indexing command and the directory from which it must be run.]

[TODO: Insert your tested VectorRAG ingestion command, including its working directory and input location.]

Both systems should use the same source corpus for a controlled retrieval comparison. If their corpora differ, document that difference and interpret the results accordingly.

### 5. Ask a question

[TODO: Insert one tested GraphRAG query command and one tested VectorRAG command or application launch command. Include the search method and required arguments.]

Suggested demonstration question:

> How does Rousseau connect individual freedom with political authority in The Social Contract?

### 6. Run the benchmark

[TODO: Insert the exact commands for generating answers with both systems, evaluating them, and producing the analysis. Specify question-file paths, output paths, and the order in which scripts should run.]

## Evaluation Method

**Question set:** [TODO: Number of questions, where the question file is stored, and how the questions were written.]

Possible categories include:

- **Specific questions:** Ask about a definition or identifiable passage.
- **Within-work synthesis:** Connect several passages in one work.
- **Cross-work synthesis:** Compare themes across multiple works.

[TODO: Replace these categories with those actually used.]

**Answer evaluation:** [TODO: Explain the rubric, scoring scale, and whether assessment used human review, an LLM judge, or both. If using an LLM judge, record its model, prompt, access to source evidence, and whether system identities were hidden.]

Suggested dimensions are relevance, coverage, and support from the source texts. A judge's preference alone does not establish factual accuracy.

**Timing:** [TODO: Explain what the timer measures, number of repeated runs, cache behavior, and treatment of failures. Separate one-time indexing from per-question response time.]

**Comparison controls:** [TODO: State whether the source texts, answer model, question set, answer-length instructions, and evidence requirements were held constant. Document differences in context budgets or retrieval settings.]

## Results

**Benchmark findings have not been filled into this template. No superiority claim is made here.**

[TODO: Replace the table with actual measurements. Add score ranges to every quality metric, report the number of evaluated questions, and link to the supporting results files.]

| Measure | VectorRAG | GraphRAG |
| --- | --- | --- |
| Questions successfully answered / attempted | [TODO] | [TODO] |
| Mean relevance score — [TODO: scale] | [TODO] | [TODO] |
| Mean coverage score — [TODO: scale] | [TODO] | [TODO] |
| Mean source-support score — [TODO: scale] | [TODO] | [TODO] |
| Median response time, seconds | [TODO] | [TODO] |
| Indexing time, minutes, if measured | [TODO] | [TODO] |
| API cost, USD, if measured — [TODO: specify scope] | [TODO] | [TODO] |

**Interpretation:** [TODO: Explain where each system performed well, where it struggled, and whether any quality improvement justified additional time or cost. If possible, separate results by question category.]

## Example Comparison

**Question:** [TODO: Copy an actual benchmark question.]

**VectorRAG answer:**

> [TODO: Insert the actual answer, preserving citations. Label it as an excerpt if shortened.]

**GraphRAG answer:**

> [TODO: Insert the actual answer, preserving citations. Label it as an excerpt if shortened.]

**Assessment:** [TODO: Explain one concrete strength or weakness in each answer, checking against the source text. Avoid selecting an example solely because it favors one system.]

## Limitations

[TODO: Retain and adapt the limitations that apply.]

- Findings from this Rousseau corpus may not generalize to other authors or domains.
- Translation choices and text preparation can affect retrieval and interpretation.
- Extracted graph relationships and generated summaries can contain errors.
- A small or manually written question set may favor particular question types.
- LLM-based evaluation can favor longer or more polished answers and may miss unsupported claims.
- Different corpora, answer models, or context budgets limit conclusions about retrieval alone.
- Response times vary with API conditions, caching, and model configuration.

## Future Work

[TODO: Keep only improvements you intend to pursue.]

- Expand the benchmark with more questions and independently checked source passages.
- Add human review of a sample of answers and citations.
- Compare additional GraphRAG search methods.
- Analyze response quality alongside latency and API cost.
- Add a simple interface for exploring questions and supporting evidence.

## References and Attribution

This project uses Microsoft's GraphRAG framework. The framework and underlying research belong to their respective authors; this repository documents my application and evaluation of the approach.


**Research Paper**
https://arxiv.org/abs/2404.16130

**Github Repository**
https://github.com/microsoft/graphrag

For VectorRAG framework, check the link below:

**Research Paper**
https://arxiv.org/abs/2005.11401


## Author

Killian K. Williams

[TODO: Add portfolio and GitHub profile links.]

## License

[TODO: Choose a license for your own code and add the corresponding LICENSE file. Third-party software and source texts retain their respective licenses and usage conditions.]
