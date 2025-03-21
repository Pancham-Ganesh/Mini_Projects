const express = require('express');
const fs = require('fs');
const csv = require('csv-parser');
const { OpenAI } = require("openai");  // Import the OpenAI SDK
const app = express();
const port = 3000;

// Serve static files (frontend)
app.use(express.static('../frontend'));

// Path to the CSV file
const trainingDataPath = 'training_data.csv';

// In-memory storage for training data
let trainingData = {};

// Function to load CSV data asynchronously
function loadTrainingData() {
    return new Promise((resolve, reject) => {
        const tempData = {};
        fs.createReadStream(trainingDataPath)
            .pipe(csv())
            .on('data', (row) => {
                // Check if 'question' and 'answer' columns exist
                if (row.question && row.answer) {
                    const question = row['question'].toLowerCase().trim();
                    const answer = row['answer'].trim();
                    tempData[question] = answer;
                } else {
                    console.warn('Skipping invalid row:', row);
                }
            })
            .on('end', () => {
                trainingData = tempData; // Store loaded data
                resolve();
            })
            .on('error', (err) => reject(err));
    });
}

// OpenAI API setup (for version 3.x.x)
const openai = new OpenAI({
  apiKey: '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',
});

// Function to fetch response from ChatGPT
async function getChatGptResponse(userMessage) {
    try {
        const response = await openai.chat.completions.create({
            model: "gpt-3.5-turbo", // You can use "gpt-4" or other models if preferred
            messages: [{ role: "user", content: userMessage }],
        });
        return response.choices[0].message.content;
    } catch (error) {
        console.error("Error with OpenAI API:", error);
        return "I'm sorry, I encountered an error while processing your request.";
    }
}

// Route to get chatbot responses
app.get('/get-response', async (req, res) => {
    if (Object.keys(trainingData).length === 0) {
        await loadTrainingData();  // Wait for training data to be loaded
    }

    const userMessage = req.query.message.toLowerCase();
    
    // Check if the question is in the training data
    if (trainingData[userMessage]) {
        // Return the response from training data if available
        return res.json({ response: trainingData[userMessage] });
    }

    // If not in training data, ask ChatGPT
    const chatGptResponse = await getChatGptResponse(userMessage);
    res.json({ response: chatGptResponse });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
