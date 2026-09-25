#include "rag_common.hpp"

int main() {
    std::string content = readFile("data/sample.txt");
    std::vector<std::string> chunks = chunkText(content, 200, 50);

    std::cout << "Loaded file: data/sample.txt (" << content.size() << " characters)\n\n";
    std::cout << "Split into " << chunks.size() << " chunks:\n\n";

    for (size_t i = 0; i < chunks.size(); i++) {
        std::cout << "--- Chunk " << i + 1 << " ---\n";
        std::cout << chunks[i] << "\n\n";
    }

    return 0;
}