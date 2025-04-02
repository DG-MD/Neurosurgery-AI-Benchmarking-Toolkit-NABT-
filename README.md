# --- Project & Citation Information ---
This repository contains the source code for the **Neurosurgery AI Benchmarking Toolkit (NABT)**. This toolkit was developed for the research project titled:

**"The Neurosurgeon’s Guide to Open-Source Artificial Intelligence: A Downloadable Toolkit for Benchmarking of Open-Source Large Language Models"**

The project was submitted to the *Journal of Neurosurgery (JNS) Focus Call for Papers*.

PROJECT_TITLE = "The Neurosurgeon’s Introduction to Open-Source Artificial Intelligence: A Downloadable Pipeline for Neurosurgical Benchmarking of Open-Source and Cloud-Based Large Language Models"
AUTHORS = "David Gomez, BS¹; Ishan Shah, BS¹; Richard Hislop, BS²; Benjamin Hopkins, MD¹; Gage Guerrera, BS¹; David J. Cote, MD PhD¹; Lawrance K. Chung MD¹; Robert G. Briggs, MD¹; William Mack MD¹; Gabriel Zada, MD¹"
AFFILIATIONS = "¹Department of Neurosurgery, Keck School of Medicine, University of Southern California, Los Angeles, CA"
JOURNAL_TARGET = "JNS Focus 2025"
CORRESPONDING_AUTHOR = "David Gomez, BS"
CORRESPONDING_EMAIL = "gomezdav@usc.edu"
---

## Project Overview

NABT is a Python-based, interactive command-line toolkit designed to provide a reproducible and modifiable framework for evaluating Large Language Models (LLMs) on neurosurgery-specific tasks. It supports benchmarking of:

1.  **Locally Stored Models:** Quantized open-source LLMs (GGUF format) run directly on user hardware (CPU/GPU).
2.  **Cloud-Based Models:** Proprietary and open-source LLMs accessed via APIs (e.g., through OpenRouter).
3.  
The toolkit automates evaluation using user-defined datasets (like the CNS SANS questions used in the paper), measuring metrics such as accuracy and inference time, and includes features for manual review and demonstration of potential clinical use cases (e.g., data extraction).

The primary goal is to empower neurosurgeons to assess, compare, and potentially implement diverse LLMs in research and clinical workflows, emphasizing privacy (via local models) and transparency.

