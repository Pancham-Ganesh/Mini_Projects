const AWS = require("aws-sdk");
const { v4: uuidv4 } = require("uuid"); // Correct way to import UUID in CommonJS
const dotenv = require("dotenv");

// Load environment variables
dotenv.config();

// Configure AWS S3
const s3 = new AWS.S3({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION,
});

// Upload file fun

dotenv.config();

// Initialize S3 client
const S3 = new AWS.S3({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION,
});

const bucketName = process.env.BUCKET_NAME;

const uploadFile = async (req, res) => {
  try {
    const fileType = req.file.mimetype; // File type (e.g., "image/png")
    const buffer = req.file.buffer; // File data

    // Generate unique file key
    const fileKey = `${uuidv4()}`;

    // S3 upload parameters
    const params = {
      Bucket: bucketName,
      Key: fileKey,
      Body: buffer,
      ContentType: fileType,
    };

    // Upload file to S3
    const data = await S3.upload(params).promise();
    const imageUrl = data.Location; // Public URL of the uploaded image

    res.status(200).json({ url: imageUrl });
  } catch (error) {
    console.error("Error uploading image to S3:", error);
    res.status(500).json({ message: "Image upload failed" });
  }
};

// Use ES Module export syntax
module.exports = uploadFile;
