const multer = require("multer");

const storage = multer.memoryStorage(); // Store file in memory buffer

const upload = multer({ storage }); // Configure multer

module.exports =  upload; // Export the multer middleware as default
