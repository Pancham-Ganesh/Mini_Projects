const formidable = require('formidable');
const fs = require('fs');
const path = require('path');
const Post = require('../models/post-model'); // MongoDB schema

const createPost = async (req, res) => {
  const { text, userId, username, imageUrl } = req.body; // Expect imageUrl to come from the frontend

  // Check if required fields are provided
  if (!text || !userId || !username || !imageUrl) {
    return res
      .status(400)
      .json({
        message: "All fields are required (text, userId, username, imageUrl)",
      });
  }

  try {
    // Create a new post with the provided data
    const post = new Post({
      text,
      userId,
      username,
      imageUrl, // Store the image URL received from the frontend
    });

    await post.save();
    res.status(201).json({ message: "Post created successfully!", post });
  } catch (error) {
    console.error("Database Error:", error);
    res.status(500).json({ message: "Internal server error" });
  }
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

    console.log("user posts")

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

    console.log("all posts");

    res.status(200).json({ posts });
  } catch (error) {
    console.error('Database Error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
};



module.exports = { createPost, getPostsByUser, getPosts };
