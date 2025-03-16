const mongoose = require("mongoose");

const postSchema = new mongoose.Schema(
  {
    text: { type: String, required: true },
    userId: { type: String, required: true },
    username: { type: String, required: true },
    imageUrl: { type: String, required: true }, // Changed to store image URL
  },
  { timestamps: true }
);

module.exports = mongoose.model("Post", postSchema);
