// UPDATED askQuestion function with STOP capability
// Replace the existing askQuestion function in nexusmind-dashboard.html with this:

let currentAbortController = null;

window.askQuestion = async function () {
    const query = queryInput.value.trim();

    if (!query) return;

    if (uploadedFiles.length === 0) {
        alert('Please upload some PDFs first!');
        return;
    }

    // Add user message
    addChatMessage(query, true);
    queryInput.value = '';

    // Create abort controller for cancellation
    currentAbortController = new AbortController();

    // Change button to STOP button
    askBtn.disabled = false;
    askBtn.onclick = stopGeneration;
    askBtn.className = 'px-8 py-4 rounded-full font-bold bg-red-500 hover:bg-red-600 transition flex items-center space-x-2';
    askBtn.innerHTML = '<i data-lucide="square" style="width: 20px; height: 20px;"></i><span>Stop</span>';
    lucide.createIcons();

    // Create AI message placeholder for streaming
    const aiMessageDiv = document.createElement('div');
    aiMessageDiv.className = 'flex justify-start';
    aiMessageDiv.innerHTML = `
        <div class="bg-white/5 border-white/10 border rounded-2xl p-4 max-w-3xl">
            <div class="flex items-start space-x-3">
                <i data-lucide="brain" class="text-blue-400" style="width: 20px; height: 20px;"></i>
                <div class="flex-1">
                    <p class="text-sm text-gray-200" id="streaming-response"><span class="animate-pulse">●</span> Thinking...</p>
                </div>
            </div>
        </div>
    `;
    chatHistory.appendChild(aiMessageDiv);
    lucide.createIcons();
    chatHistory.scrollTop = chatHistory.scrollHeight;

    const responseElement = document.getElementById('streaming-response');

    try {
        const response = await fetch(`${API_BASE_URL}/query-stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                userId: userId,
                topK: 5
            }),
            signal: currentAbortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let sources = null;

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);

                    if (data === '[DONE]') {
                        break;
                    }

                    try {
                        const parsed = JSON.parse(data);

                        if (parsed.token) {
                            fullResponse += parsed.token;
                            responseElement.textContent = fullResponse;
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        } else if (parsed.sources) {
                            sources = parsed.sources;
                        } else if (parsed.error) {
                            responseElement.textContent = '❌ ' + parsed.error;
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete chunks
                    }
                }
            }
        }

        // Add sources if available
        if (sources && sources.length > 0) {
            addSourcesCitation(sources);
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            responseElement.textContent = '⏹️ Generation stopped by user.';
        } else {
            console.error('Query error:', error);
            responseElement.textContent = '❌ Error: ' + error.message;
        }
    } finally {
        // Reset button back to ASK
        askBtn.disabled = false;
        askBtn.onclick = askQuestion;
        askBtn.className = 'px-8 py-4 rounded-full font-bold bg-gradient-to-r from-electric-violet to-blue-500 hover:opacity-90 transition flex items-center space-x-2';
        askBtn.innerHTML = '<i data-lucide="send" style="width: 20px; height: 20px;"></i><span>Ask</span>';
        lucide.createIcons();
        currentAbortController = null;
    }
};

function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
}
