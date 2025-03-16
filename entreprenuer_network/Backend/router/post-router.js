const express = require('express');
const router = express.Router();

const { createPost ,getPostsByUser, getPosts } = require('../controllers/postController'); // Controller we'll define
const authMiddleware = require("../middlewares/authMiddleware.js")

router.route('/uploadpost').post(authMiddleware, createPost); // Route for creating a post

router.route('/getposts/user/:userId').get(authMiddleware, getPostsByUser);

router.route('/getposts/user').get(authMiddleware, getPosts);

module.exports = router;