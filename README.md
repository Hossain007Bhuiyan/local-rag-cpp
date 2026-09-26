# Local RAG in C++

A small AI chat tool. It reads your text files, finds the most relevant part and asks a local LLM to answer using that part.

Built with [llama.cpp](https://github.com/ggml-org/llama.cpp) for running the models.
<img width="1204" height="761" alt="chat" src="https://github.com/user-attachments/assets/587e8ed9-69f1-4384-a52f-c4b7a250f829" />


## What it does

1. Reads every `.txt` file in the `data` folder
2. Splits the text into small chunks
3. Turns each chunk into a vector (an embedding) so it can be searched by meaning, not just keywords
4. When you ask a question, it finds the chunks that match best
5. Sends those chunks plus your question to a small local LLM
6. Shows you the answer, plus which file it came from

## Why build this

Most RAG tutorials use Python. This one is written in C++ from the ground up. The goal was to learn how the pieces actually work: chunking, embeddings, vector search and prompting a local model.

## Two ways to use it

**Browser chat**

Opens a chat window in your browser.

```bash
./chat_gui
```

**Terminal chat**

Same thing but you type in the Terminal instead.

```bash
./chat
```

Type `exit` to quit.

## Setup

You need a Mac with Xcode Command Line Tools, CMake, and Homebrew.

1. Clone this repo
2. Clone and build llama.cpp inside this folder:
```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   cmake -B build -DLLAMA_OPENSSL=ON -DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3)
   cmake --build build --config Release -j
   cd ..
```
3. Build the project:
```bash
   chmod +x build.sh
   ./build.sh
```
4. Add your own `.txt` files to the `data` folder
5. Run it:
```bash
   ./chat_gui
```

The first run will download two small models automatically (an embedding model and a chat model).

## Project files

| File | What it does |
|---|---|
| `rag_common.hpp` | shared code: reading files, chunking, embeddings, search |
| `main.cpp` | basic demo, just chunking |
| `embeddings.cpp` | demo of chunking + search, no chat |
| `rag.cpp` | full pipeline, one fixed question |
| `chat.cpp` | full pipeline, terminal chat |
| `chat_gui.cpp` | full pipeline, browser chat window |

## Known limits

- Only reads `.txt` files right now
- Each question re-runs the model fresh, so it's not instant
- Small local models can sometimes ramble or repeat themselves

## What I'd add next

- PDF support
- A faster setup that keeps the model loaded between questions
- Better handling of longer documents
