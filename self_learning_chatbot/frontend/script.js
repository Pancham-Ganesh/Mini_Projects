const chatLog = document.getElementById("chat-log");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

sendBtn.addEventListener("click", function () {
  sendMessage();
});

// Allow sending message by pressing Enter key
userInput.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});

function sendMessage() {
  const userMessage = userInput.value.trim();

  if (userMessage !== "") {
    // Add user message to chat log
    chatLog.innerHTML += `<div><strong>You:</strong> ${userMessage}</div>`;
    userInput.value = ""; // Clear input field

    // Show "Bot is typing..." indicator
    chatLog.innerHTML += `<div id="typing-indicator"><em>Bot is typing...</em></div>`;

    // Scroll to bottom
    chatLog.scrollTop = chatLog.scrollHeight;

    // Send user message to backend and get response
    fetch(`/get-response?message=${encodeURIComponent(userMessage)}`)
      .then((response) => response.json())
      .then((data) => {
        const botResponse = data.response;

        // Remove "Bot is typing..." indicator
        document.getElementById("typing-indicator").remove();

        // Add bot response to chat log
        chatLog.innerHTML += `<div><strong>Bot:</strong> ${botResponse}</div>`;
        chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
      })
      .catch((error) => {
        console.error("Error:", error);
        document.getElementById("typing-indicator").remove();
        chatLog.innerHTML += `<div style="color: red;"><strong>Bot:</strong> Error processing request. Please try again.</div>`;
      });

    // Focus back on input field
    userInput.focus();
  }
}
