import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  TouchableWithoutFeedback,
  Keyboard,
  SafeAreaView,
} from "react-native";

export default function App() {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  const sendMessage = async () => {
    if (!message.trim()) return;

    try {
      const res = await fetch(
        `http://192.168.31.85:3000/get-response?message=${encodeURIComponent(
          message
        )}`
      );
      const data = await res.json();
      setChatHistory([...chatHistory, { user: message, bot: data.response }]);
      setMessage(""); // Clear input field
    } catch (error) {
      setChatHistory([
        ...chatHistory,
        { user: message, bot: "Error connecting to server." },
      ]);
      console.error(error);
    }
  };

  const clearChat = () => {
    setChatHistory([]);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.container}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <View style={styles.innerContainer}>
            {/* Header */}
            <View style={styles.header}>
              <Text style={styles.headerText}>WorkWise AI</Text>
              <TouchableOpacity style={styles.clearButton} onPress={clearChat}>
                <Text style={styles.clearButtonText}>Clear Chat</Text>
              </TouchableOpacity>
            </View>

            {/* Chat Area */}
            <ScrollView style={styles.chatArea}>
              {chatHistory.map((chat, index) => (
                <View key={index} style={styles.messageContainer}>
                  <Text style={styles.userMessage}>You: {chat.user}</Text>
                  <Text style={styles.botMessage}>Bot: {chat.bot}</Text>
                </View>
              ))}
            </ScrollView>

            {/* Input & Send Button */}
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Type your message..."
                value={message}
                onChangeText={setMessage}
                onSubmitEditing={sendMessage} // Send on Enter (optional)
              />
              <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
                <Text style={styles.sendButtonText}>Send</Text>
              </TouchableOpacity>
            </View>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#ADD8E6", // Light blue background
  },
  container: {
    flex: 1,
  },
  innerContainer: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "#2C3E50",
    padding: 15,
  },
  headerText: {
    color: "#FFF",
    fontSize: 18,
    fontWeight: "bold",
  },
  clearButton: {
    backgroundColor: "#E74C3C",
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 5,
  },
  clearButtonText: {
    color: "#FFF",
    fontWeight: "bold",
  },
  chatArea: {
    flex: 1,
    padding: 10,
  },
  messageContainer: {
    marginBottom: 10,
  },
  userMessage: {
    backgroundColor: "#3498DB",
    color: "#FFF",
    padding: 10,
    borderRadius: 5,
    alignSelf: "flex-end",
    marginBottom: 5,
  },
  botMessage: {
    backgroundColor: "#2ECC71",
    color: "#FFF",
    padding: 10,
    borderRadius: 5,
    alignSelf: "flex-start",
  },
  inputContainer: {
    flexDirection: "row",
    padding: 10,
    backgroundColor: "#2C3E50",
    alignItems: "center",
  },
  input: {
    flex: 1,
    backgroundColor: "#FFF",
    padding: 10,
    borderRadius: 20,
    marginRight: 10,
  },
  sendButton: {
    backgroundColor: "#27AE60",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  sendButtonText: {
    color: "#FFF",
    fontWeight: "bold",
  },
});
