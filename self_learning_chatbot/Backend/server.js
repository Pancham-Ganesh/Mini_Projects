const express = require("express");
const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const { OpenAI } = require("openai");
const Fuse = require("fuse.js"); // For fuzzy matching

const app = express();
const port = 3000;

// Serve static files from the frontend directory
app.use(express.static(path.join(__dirname, "../frontend")));

// Path to the CSV file
const trainingDataPath = "training_data.csv";

// In-memory storage for training data (Exact Matches)
let trainingData = {};

// In-memory storage for structured employee data (For Fuzzy Search)
let structuredData = {};

// Function to load CSV data into memory
function loadTrainingData() {
  return new Promise((resolve, reject) => {
    const exactMatches = {}; // For exact Q&A pairs
    const tempStructuredData = {}; // For employee details

    fs.createReadStream(trainingDataPath)
      .pipe(csv())
      .on("data", (row) => {
        const question = row["question"]?.trim();
        const answer = row["answer"]?.trim();

        if (!question || !answer) {
          console.warn("Skipping invalid row:", row);
          return;
        }

        // Store for exact question matching
        exactMatches[question.toLowerCase()] = answer;

        // Extract the employee's name from the question (assume last 2 words are the name)
        const words = question.split(" ");
        let possibleName = words.slice(-2).join(" ");

        if (!tempStructuredData[possibleName])
          tempStructuredData[possibleName] = [];
        tempStructuredData[possibleName].push({ question, answer });
      })
      .on("end", () => {
        trainingData = exactMatches; // Store exact matches
        structuredData = tempStructuredData; // Store employee details
        resolve();
      })
      .on("error", (err) => reject(err));
  });
}

// OpenAI API setup
const openai = new OpenAI({
  apiKey:
    "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", // Replace with a valid API key
});

// Function to fetch response from ChatGPT
async function getChatGptResponse(userMessage) {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4-turbo",
      messages: [{ role: "user", content: userMessage }],
    });
    return response.choices[0].message.content;
  } catch (error) {
    console.error("Error with OpenAI API:", error);
    return "I'm sorry, I encountered an error while processing your request.";
  }
}

// Route to get chatbot responses
app.get("/get-response", async (req, res) => {
  if (Object.keys(trainingData).length === 0) {
    await loadTrainingData(); // Load data if not already loaded
  }

  const userMessage = req.query.message.toLowerCase().trim();

  // **Step 1: Check for an exact match first**
  if (trainingData[userMessage]) {
    return res.json({ response: trainingData[userMessage] });
  }

  // **Step 2: Try to match the message to an employee name**
  const fuse = new Fuse(Object.keys(structuredData), { threshold: 0.4 });
  const result = fuse.search(userMessage);

  if (result.length > 0) {
    const matchedName = result[0].item;
    const employeeData = structuredData[matchedName];

    // **Format the structured response**
    const responseText = employeeData
      .map((entry) => `- **${entry.question}**: ${entry.answer}`)
      .join("\n");

    return res.json({
      response: `Here are the details for **${matchedName}**:\n\n${responseText}`,
    });
  }

  // **Step 3: If no exact or fuzzy match, use OpenAI**
  const chatGptResponse = await getChatGptResponse(userMessage);
  res.json({ response: chatGptResponse });
});

// Serve frontend (index.html) when accessing the root URL
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/index.html"));
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
