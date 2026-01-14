# Langchain Pipelines

A comprehensive collection of LangChain tutorials and examples covering the fundamental concepts and components of building LLM-powered applications.

## 📚 Table of Contents

- [About](#about)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Modules Overview](#modules-overview)
- [Usage](#usage)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

## 🎯 About

This repository contains hands-on examples and tutorials for learning LangChain, a powerful framework for developing applications powered by language models. Each module focuses on a specific aspect of LangChain, providing practical examples through Jupyter notebooks.

## 📂 Repository Structure

```
Langchain-pipelines/
│
├── 1.langchain_models/           # Working with different LLM models
├── 2.langchain_prompts/          # Creating and managing prompt templates
├── 3.langchain_structured_output/# Handling structured outputs from LLMs
├── 4.langchain_output_parser/    # Parsing and processing LLM outputs
├── 5.langchain_chains/           # Building chains of LLM operations
├── 6.langchain_runnables/        # Using the Runnable interface
├── 7.langchain_document_loader/  # Loading documents from various sources
├── 8.langchain_text_splitter/    # Splitting text for processing
├── 9.langchain_vector_store/     # Working with vector databases
├── 10.langchain_retrievers/      # Implementing retrieval mechanisms
├── 11.langchain_tools/           # Integrating external tools with LLMs
├── langchain_notes.pdf           # Supplementary documentation
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- OpenAI API key (or other LLM provider credentials)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Devatva24/Langchain-pipelines.git
cd Langchain-pipelines
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install langchain langchain-openai jupyter notebook
pip install chromadb tiktoken  # For vector store examples
```

4. Set up your API keys:
```bash
export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
```

## 📖 Modules Overview

### 1. LangChain Models
Learn how to work with different language models including OpenAI, Anthropic, and open-source alternatives.

### 2. LangChain Prompts
Master the art of creating reusable and dynamic prompt templates for better LLM interactions.

### 3. Structured Output
Understand how to get structured, predictable outputs from language models.

### 4. Output Parser
Learn techniques for parsing and validating LLM responses into usable data structures.

### 5. LangChain Chains
Build complex workflows by chaining multiple LLM calls and operations together.

### 6. Runnables
Explore the Runnable interface for creating composable and streamable LLM pipelines.

### 7. Document Loader
Load and process documents from various sources including PDFs, web pages, and databases.

### 8. Text Splitter
Learn strategies for splitting large documents into manageable chunks for processing.

### 9. Vector Store
Work with vector databases for semantic search and retrieval-augmented generation (RAG).

### 10. Retrievers
Implement different retrieval strategies for finding relevant information from your data.

### 11. Tools
Integrate external tools and APIs to extend LLM capabilities with real-world actions.

## 💻 Usage

Navigate to any module directory and open the Jupyter notebooks:

```bash
cd 1.langchain_models
jupyter notebook
```

Each notebook contains:
- Theoretical explanation
- Practical code examples
- Hands-on exercises
- Best practices and tips

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- Additional notes: See `langchain_notes.pdf`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions or feedback, please open an issue in this repository.

---

**Happy Learning!** 🎉
