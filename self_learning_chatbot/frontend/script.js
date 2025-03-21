const chatLog = document.getElementById('chat-log');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

sendBtn.addEventListener('click', function() {
    const userMessage = userInput.value.trim();

    if (userMessage !== "") {
        // Add user message to chat log
        chatLog.innerHTML += `<div><strong>You:</strong> ${userMessage}</div>`;
        userInput.value = ''; // Clear input field

        // Scroll to bottom
        chatLog.scrollTop = chatLog.scrollHeight;

        // Send user message to backend and get response
        fetch(`/get-response?message=${encodeURIComponent(userMessage)}`)
            .then(response => response.json())
            .then(data => {
                const botResponse = data.response;

                // Add bot response to chat log
                chatLog.innerHTML += `<div><strong>Bot:</strong> ${botResponse}</div>`;
                chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    // Focus back on input field
    userInput.focus();
});
