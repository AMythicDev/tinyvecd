# tinyvecd

An extremely simple vector database implementation tailored for RAG

- Only tailored for RAG purposes, nothing else is supported.
- Implements the HNSW algorithm with cosine similarity as parameter for finding closest documents.
- A hybrid custom flat-file + sqlite database storage solution to store embeddings and documents metadata respectively.
- The embeddings file is memory mapped for direct reading-writing avoiding syscalls.
- Most operations on the embeddings file is done in zero copy fashion avoiding unnecessary allocations.
- Integrate with the kernel's filesystem notification subsystem to embed and delete embeddings as files are added or removed.
- Support for reconciling the database after restart.
- Cosine similarity is optimized for x86_64 leveraging AVX-256 if supported.

NOTE:
- Only tested on UNIX-like systems. It might work on Windows however it is not a guarantee.
- The main.rs file is only a reference usage of the library not an actual full featured thing to run in production.
- A custom Gemini embedding provider is already provided for reference.

