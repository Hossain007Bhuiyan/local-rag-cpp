#!/bin/bash
set -e

clang++ -std=c++17 src/main.cpp -o chunker
clang++ -std=c++17 src/embeddings.cpp -o embedder
clang++ -std=c++17 src/rag.cpp -o rag
clang++ -std=c++17 src/chat.cpp -o chat

clang++ -std=c++17 -pthread src/chat_gui.cpp \
  llama.cpp/build/vendor/cpp-httplib/libcpp-httplib.a \
  -L$(brew --prefix openssl@3)/lib -lssl -lcrypto \
  -framework CoreFoundation -framework Security \
  -o chat_gui

echo "All built."