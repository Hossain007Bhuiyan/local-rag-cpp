#include "rag_common.hpp"
#include "../llama.cpp/vendor/cpp-httplib/httplib.h"

// escape for JSON
std::string jsonEscape(const std::string& text) {
    std::string out;
    for (char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

// naive json field extraction, good enough here
std::string parseJsonField(const std::string& body, const std::string& field) {
    std::string marker = "\"" + field + "\":\"";
    size_t start = body.find(marker);
    if (start == std::string::npos) return "";
    start += marker.size();

    std::string result;
    for (size_t i = start; i < body.size(); i++) {
        char c = body[i];
        if (c == '\\' && i + 1 < body.size()) {
            char next = body[i + 1];
            if (next == 'n') { result += '\n'; i++; continue; }
            if (next == '"') { result += '"'; i++; continue; }
            if (next == '\\') { result += '\\'; i++; continue; }
        }
        if (c == '"') break;
        result += c;
    }
    return result;
}

const char* CHAT_PAGE = R"HTML(
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Local RAG Chat</title>
<style>
  body { margin: 0; font-family: -apple-system, sans-serif; background: #1e1e1e; color: #eee; }
  #chat { max-width: 700px; margin: 0 auto; height: 100vh; display: flex; flex-direction: column; }
  #messages { flex: 1; overflow-y: auto; padding: 20px; }
  .msg { margin-bottom: 16px; line-height: 1.4; }
  .you { color: #7fb8ff; }
  .ai { color: #ececec; }
  .source { font-size: 12px; opacity: 0.5; margin-top: 4px; }
  .label { font-size: 12px; opacity: 0.6; margin-bottom: 4px; }
  #inputBar { display: flex; padding: 16px; border-top: 1px solid #333; background: #252525; }
  #question { flex: 1; padding: 10px; border-radius: 8px; border: 1px solid #444;
              background: #1e1e1e; color: #eee; font-size: 14px; }
  #send { margin-left: 8px; padding: 10px 18px; border-radius: 8px; border: none;
          background: #4a7dff; color: white; cursor: pointer; }
  #send:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<div id="chat">
  <div id="messages"></div>
  <div id="inputBar">
    <input id="question" type="text" placeholder="Ask a question about your documents..." autofocus>
    <button id="send">Send</button>
  </div>
</div>
<script>
  const messages = document.getElementById('messages');
  const input = document.getElementById('question');
  const button = document.getElementById('send');

  function addMessage(text, who, sources) {
    const div = document.createElement('div');
    div.className = 'msg';
    let html = '<div class="label">' + (who === 'you' ? 'You' : 'AI') + '</div>' +
               '<div class="' + who + '">' + text + '</div>';
    if (sources) html += '<div class="source">Source: ' + sources + '</div>';
    div.innerHTML = html;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, 'you');
    input.value = '';
    input.disabled = true;
    button.disabled = true;
    const thinking = addMessage('Thinking...', 'ai');

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text })
      });
      const data = await res.json();
      thinking.querySelector('.ai').textContent = data.answer;
      if (data.sources) {
        const src = document.createElement('div');
        src.className = 'source';
        src.textContent = 'Source: ' + data.sources;
        thinking.appendChild(src);
      }
    } catch (err) {
      thinking.querySelector('.ai').textContent = 'Something went wrong: ' + err;
    }

    input.disabled = false;
    button.disabled = false;
    input.focus();
  }

  button.addEventListener('click', send);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>
</body>
</html>
)HTML";

int main() {
    auto chunks = loadAllDocuments("data");

    std::cout << "Loaded " << chunks.size() << " chunks. Embedding them, one moment...\n";

    std::vector<std::vector<float>> embeddings;
    for (const auto& chunk : chunks) {
        embeddings.push_back(getEmbedding(chunk.text));
    }

    std::cout << "Ready. Starting server...\n";

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(CHAT_PAGE, "text/html");
    });

    svr.Post("/ask", [&](const httplib::Request& req, httplib::Response& res) {
        std::string question = parseJsonField(req.body, "question");

        std::vector<float> qEmbedding = getEmbedding(question);
        auto topMatches = findTopMatches(qEmbedding, embeddings, 3);

        std::string context;
        std::string sources;
        for (size_t i = 0; i < topMatches.size(); i++) {
            context += chunks[topMatches[i]].text + "\n\n";
            sources += chunks[topMatches[i]].sourceFile;
            if (i + 1 < topMatches.size()) sources += ", ";
        }

        std::string answer = askLLM(context, question);

        std::string json = "{\"answer\":\"" + jsonEscape(answer) +
                            "\",\"sources\":\"" + jsonEscape(sources) + "\"}";
        res.set_content(json, "application/json");
    });

    std::cout << "Open in your browser: http://127.0.0.1:8080\n";
    system("open http://127.0.0.1:8080");

    svr.listen("127.0.0.1", 8080);

    return 0;
}