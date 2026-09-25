#include "rag_common.hpp"

int main() {
    auto chunks = loadAllDocuments("data");

    std::cout << "Loaded " << chunks.size() << " chunks. Embedding them, one moment...\n";

    std::vector<std::vector<float>> embeddings;
    for (const auto& chunk : chunks) {
        embeddings.push_back(getEmbedding(chunk.text));
    }

    std::cout << "Ready. Ask a question (or type 'exit' to quit).\n\n";

    std::string question;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, question);

        if (question == "exit" || question == "quit") break;
        if (trim(question).empty()) continue;

        std::vector<float> qEmbedding = getEmbedding(question);
        auto topMatches = findTopMatches(qEmbedding, embeddings, 3);

        std::string context;
        for (size_t idx : topMatches) {
            context += chunks[idx].text + "\n\n";
        }

        std::string answer = askLLM(context, question);
        std::cout << "AI: " << answer << "\n";
        std::cout << "(Sources: ";
        for (size_t i = 0; i < topMatches.size(); i++) {
            std::cout << chunks[topMatches[i]].sourceFile;
            if (i + 1 < topMatches.size()) std::cout << ", ";
        }
        std::cout << ")\n\n";
    }

    std::cout << "Goodbye.\n";
    return 0;
}