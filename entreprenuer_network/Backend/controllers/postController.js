const formidable = require('formidable');
const fs = require('fs');
const path = require('path');
const Post = require('../models/post-model'); // MongoDB schema

const createPost = (req, res) => {
  const form = new formidable.IncomingForm();
  form.uploadDir = path.join(__dirname, '../uploads');
  form.keepExtensions = true;
  form.parse(req, async (err, fields, files) => {
    if (err) {
      console.error('Form Parsing Error:', err);
      return res.status(400).json({ message: 'Error parsing form data' });
    }

    console.log('Uploaded Files:', files);

    let { text, userId, username } = fields;
    const { image } = files;

    // Convert text and userId to strings if they are arrays
    if (Array.isArray(text)) {
      text = text.join(' '); // Join array elements to form a single string
    }
    if (Array.isArray(userId)) {
      userId = userId[0]; // Get the first element if it's an array
    }

    if (Array.isArray(username)) {
      username = username[0]; // Get the first element if it's an array
    }

    // Check if image is uploaded
    if (!image || !image[0]) {
      return res.status(400).json({ message: 'Image is required' });
    }

    const uploadedImage = image[0];
    const imagePath = uploadedImage.filepath;
    const imageExtension = path.extname(uploadedImage.originalFilename);

    if (!imagePath) {
      return res.status(400).json({ message: 'Image path is missing' });
    }

    try {
      const newImagePath = path.join(form.uploadDir, uploadedImage.newFilename + imageExtension);
      fs.renameSync(imagePath, newImagePath);

      const imageData = fs.readFileSync(newImagePath);
      const contentType = uploadedImage.mimetype;

      const post = new Post({
        text,
        userId,
        image: {
          data: imageData,
          contentType: contentType,
          path: newImagePath,
        },
        username,
      });

      await post.save();
      res.status(201).json({ message: 'Post created successfully!', post });
    } catch (error) {
      console.error('Database Error:', error);
      res.status(500).json({ message: 'Internal server error' });
    }
  });
};

// Fetch all posts for a particular userId
const getPostsByUser = async (req, res) => {
  const { userId } = req.params; // Get the userId from the URL params

  try {
    // Query the database to get all posts for the given userId
    const posts = await Post.find({ userId });

    if (posts.length === 0) {
      return res.status(404).json({ message: 'No posts found for this user' });
    }

    res.status(200).json({ posts });
  } catch (error) {
    console.error('Database Error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
};

// Fetch all posts from the database
const getPosts = async (req, res) => {
  try {
    // Query the database to get all posts
    const posts = await Post.find();

    if (posts.length === 0) {
      return res.status(404).json({ message: 'No posts found' });
    }

    res.status(200).json({ posts });
  } catch (error) {
    console.error('Database Error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
};



module.exports = { createPost, getPostsByUser, getPosts };
