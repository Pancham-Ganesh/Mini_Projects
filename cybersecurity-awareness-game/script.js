// Questions Array
const questions = [
    {
      question: "What is a strong password?",
      answers: [
        { text: "123456", correct: false },
        { text: "mypassword", correct: false },
        { text: "P@ssw0rd!123", correct: true },
        { text: "password1", correct: false },
      ],
      feedback: "A strong password includes numbers, symbols, and uppercase letters. Avoid common phrases.",
    },
    {
      question: "Which email looks suspicious?",
      answers: [
        { text: "admin@company.com", correct: false },
        { text: "support@amazon-official.com", correct: true },
        { text: "helpdesk@school.edu", correct: false },
        { text: "info@business.org", correct: false },
      ],
      feedback: "Correct! 'amazon-official.com' is a fake domain. Always double-check URLs.",
    },
    {
      question: "What should you do if you receive a suspicious link?",
      answers: [
        { text: "Click to see what it is.", correct: false },
        { text: "Ignore and delete the email.", correct: true },
        { text: "Forward it to your contacts.", correct: false },
        { text: "Reply asking for clarification.", correct: false },
      ],
      feedback: "Never click suspicious links. Ignoring and deleting is safest.",
    },
    {
      question: "What does HTTPS in a URL indicate?",
      answers: [
        { text: "The site is fast.", correct: false },
        { text: "The site is secure.", correct: true },
        { text: "The site has ads.", correct: false },
        { text: "The site has a firewall.", correct: false },
      ],
      feedback: "Correct! HTTPS ensures encrypted communication.",
    },
    {
      question: "What is phishing?",
      answers: [
        { text: "A method to steal sensitive information.", correct: true },
        { text: "A fun activity on a boat.", correct: false },
        { text: "A way to speed up the internet.", correct: false },
        { text: "A new social media trend.", correct: false },
      ],
      feedback: "Phishing attempts to steal data like passwords. Be cautious!",
    },
  ];
  
  // DOM Elements
  const landingPage = document.getElementById("landing-page");
  const gameContainer = document.getElementById("game-container");
  const startGameBtn = document.getElementById("start-game-btn");
  const questionText = document.getElementById("question-text");
  const answerButtons = document.querySelectorAll(".answer-btn");
  const feedbackSection = document.getElementById("feedback-section");
  const feedbackText = document.getElementById("feedback-text");
  const timerDisplay = document.getElementById("time-left");
  const finalScoreDisplay = document.getElementById("final-score");
  const scoreSection = document.getElementById("score-section");
  const restartButton = document.getElementById("restart-btn");
  
  let currentQuestionIndex = 0;
  let score = 0;
  let timeLeft = 30;
  let timerInterval;
  
  // Start Game
  startGameBtn.addEventListener("click", () => {
    landingPage.style.display = "none";
    gameContainer.hidden = false;
    startGame();
  });
  
  // Start the game
  function startGame() {
    currentQuestionIndex = 0;
    score = 0;
    timeLeft = 30;
    scoreSection.style.display = "none"; // Hide score section
    document.getElementById("question-section").style.display = "block"; // Show questions
    showQuestion();
    startTimer();
  }
  
  // Show the current question
  function showQuestion() {
    const currentQuestion = questions[currentQuestionIndex];
    questionText.textContent = currentQuestion.question;
  
    answerButtons.forEach((button, index) => {
      const answer = currentQuestion.answers[index];
      button.textContent = answer.text;
      button.onclick = () => checkAnswer(answer.correct, currentQuestion.feedback);
    });
  }
  
  // Check if the selected answer is correct
  function checkAnswer(isCorrect, feedback) {
    clearInterval(timerInterval); // Stop the timer
    feedbackText.textContent = feedback;
    feedbackSection.style.display = "block"; // Show feedback
  
    if (isCorrect) {
      score++; // Increment score if the answer is correct
    }
  
    // Proceed to next question after 3 seconds
    setTimeout(() => {
      feedbackSection.style.display = "none";
      nextQuestion();
    }, 3000);
  }
  
  // Move to the next question
  function nextQuestion() {
    currentQuestionIndex++;
    if (currentQuestionIndex < questions.length) {
      timeLeft = 30; // Reset timer for next question
      showQuestion();
      startTimer();
    } else {
      endGame(); // No more questions, end the game.
    }
  }
  
  // Timer logic
  function startTimer() {
    clearInterval(timerInterval); // Ensure no duplicate timers
    timerInterval = setInterval(() => {
      timeLeft--;
      timerDisplay.textContent = timeLeft;
  
      if (timeLeft <= 0) {
        clearInterval(timerInterval); // Stop timer when time is up
        feedbackText.textContent = "Time's up! Moving to the next question.";
        feedbackSection.style.display = "block";
  
        setTimeout(() => {
          feedbackSection.style.display = "none";
          nextQuestion();
        }, 3000);
      }
    }, 1000);
  }
  
  // End the game and show the score
  function endGame() {
    clearInterval(timerInterval); // Stop any running timer
    scoreSection.style.display = "block"; // Show score section
    finalScoreDisplay.textContent = `${score}/${questions.length}`; // Display final score
    document.getElementById("question-section").style.display = "none"; // Hide questions
  }
  
  // Restart the game
  restartButton.addEventListener("click", () => {
    location.reload(); // Reload the page to restart
  });
  