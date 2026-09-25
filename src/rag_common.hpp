#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// text + which file it came from
struct DocumentChunk {
    std::string text;
    std::string sourceFile;
};

inline std::string readFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

inline std::vector<std::string> chunkText(const std::string& text, size_t chunkSize, size_t overlap) {
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = std::min(start + chunkSize, text.size());
        chunks.push_back(text.substr(start, end - start));
        if (end == text.size()) break;
        start = end - overlap;
    }
    return chunks;
}

// loads all .txt files and chunks them
inline std::vector<DocumentChunk> loadAllDocuments(const std::string& folder, size_t chunkSize = 200, size_t overlap = 50) {
    std::vector<DocumentChunk> result;

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() != ".txt") continue;

        std::string content = readFile(entry.path().string());
        std::vector<std::string> chunks = chunkText(content, chunkSize, overlap);

        for (const auto& chunk : chunks) {
            result.push_back({ chunk, entry.path().filename().string() });
        }
    }

    return result;
}

inline std::string shellEscape(const std::string& text) {
    std::string escaped = "'";
    for (char c : text) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped += c;
        }
    }
    escaped += "'";
    return escaped;
}

inline std::string runCommand(const std::string& cmd) {
    std::array<char, 4096> buffer;
    std::string result;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen failed");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

inline std::string trim(const std::string& text) {
    size_t start = text.find_first_not_of(" \t\n\r");
    size_t end = text.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return text.substr(start, end - start + 1);
}

inline std::vector<float> parseEmbedding(const std::string& raw) {
    std::vector<float> values;

    size_t start = raw.find("\"embedding\": [");
    if (start == std::string::npos) {
        throw std::runtime_error("no embedding found in output:\n" + raw);
    }
    start += std::string("\"embedding\": [").size();

    size_t end = raw.find(']', start);
    if (end == std::string::npos) {
        throw std::runtime_error("malformed embedding array");
    }

    std::stringstream ss(raw.substr(start, end - start));
    std::string token;
    while (std::getline(ss, token, ',')) {
        values.push_back(std::stof(token));
    }

    return values;
}

inline std::vector<float> getEmbedding(const std::string& text) {
    std::string cmd = "llama.cpp/build/bin/llama-embedding -hf huoxu/all-MiniLM-L6-v2-Q4_0-GGUF "
                       "-p " + shellEscape(text) + " --embd-output-format json 2>/dev/null";
    return parseEmbedding(runCommand(cmd));
}

inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}


inline size_t findBestMatch(const std::vector<float>& questionEmbedding,
                             const std::vector<std::vector<float>>& chunkEmbeddings) {
    size_t bestIndex = 0;
    float bestScore = -1.0f;
    for (size_t i = 0; i < chunkEmbeddings.size(); i++) {
        float score = cosineSimilarity(questionEmbedding, chunkEmbeddings[i]);
        if (score > bestScore) {
            bestScore = score;
            bestIndex = i;
        }
    }
    return bestIndex;
}

// top N matches, sorted best first
inline std::vector<size_t> findTopMatches(const std::vector<float>& questionEmbedding,
                                           const std::vector<std::vector<float>>& chunkEmbeddings,
                                           size_t topN) {
    std::vector<std::pair<float, size_t>> scored;
    for (size_t i = 0; i < chunkEmbeddings.size(); i++) {
        scored.push_back({ cosineSimilarity(questionEmbedding, chunkEmbeddings[i]), i });
    }

    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<size_t> result;
    for (size_t i = 0; i < std::min(topN, scored.size()); i++) {
        result.push_back(scored[i].second);
    }
    return result;
}

inline std::string askLLM(const std::string& context, const std::string& question) {
    std::string prompt =
        "Answer the question using only the context below. "
        "Keep it short, a few sentences is fine. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n"
        "Answer:";

    std::string cmd = "llama.cpp/build/bin/llama-completion "
                       "-hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M "
                       "-p " + shellEscape(prompt) +
                       " -n 150 --temp 0.3 -no-cnv --no-display-prompt 2>/dev/null";

    return trim(runCommand(cmd));
}