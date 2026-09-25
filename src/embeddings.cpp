#include "rag_common.hpp"

int main() {
    std::string content = readFile("data/sample.txt");
    std::vector<std::string> chunks = chunkText(content, 200, 50);

    std::cout << "Embedding " << chunks.size() << " chunks...\n";

    std::vector<std::vector<float>> embeddings;
    for (const auto& chunk : chunks) {
        embeddings.push_back(getEmbedding(chunk));
    }

    std::string query = "What is RAG?";
    std::vector<float> queryEmbedding = getEmbedding(query);

    std::cout << "\nQuery: " << query << "\n\n";
    for (size_t i = 0; i < chunks.size(); i++) {
        float score = cosineSimilarity(queryEmbedding, embeddings[i]);
        std::cout << "chunk " << i + 1 << "  similarity: " << score << "\n";
    }

    return 0;
}