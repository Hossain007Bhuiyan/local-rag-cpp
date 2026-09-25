#include "rag_common.hpp"

int main() {
    auto chunks = loadAllDocuments("data");

    std::cout << "Embedding " << chunks.size() << " chunks...\n";

    std::vector<std::vector<float>> embeddings;
    for (const auto& chunk : chunks) {
        embeddings.push_back(getEmbedding(chunk.text));
    }

    std::string question = "What is RAG?";
    std::vector<float> questionEmbedding = getEmbedding(question);

    size_t bestIndex = findBestMatch(questionEmbedding, embeddings);

    std::cout << "\nBest match: " << chunks[bestIndex].sourceFile << "\n";
    std::cout << "Asking the model...\n\n";

    std::string answer = askLLM(chunks[bestIndex].text, question);

    std::cout << "Question: " << question << "\n";
    std::cout << "Answer: " << answer << "\n";
    std::cout << "(Source: " << chunks[bestIndex].sourceFile << ")\n";

    return 0;
}