const jwt = require("jsonwebtoken");
const User = require("../models/user-model");

const authMiddleware = async (req, res, next) => {
  // Get the token from the Authorization header
  const token = req.header("Authorization");

  // Check if the token exists
  if (!token) {
    return res
      .status(401)
      .json({ message: "Unauthorized HTTP, Token not provided" });
  }

  // Extract the token from the "Bearer <token>" format
  const jwtToken = token.replace("Bearer", "").trim();

  try {
    // Verify the token
    const isVerified = jwt.verify(jwtToken, process.env.JWT_SECRET_KEY);

    // Fetch user data from the database (excluding the password)
    const userData = await User.findOne({ email: isVerified.email }).select({
      password: 0,
    });

    // Check if the user exists
    if (!userData) {
      return res.status(404).json({ message: "User not found" });
    }

    // Attach user data, token, and userId to the request object
    req.user = userData;
    req.token = token;
    req.userId = userData._id;

    // Proceed to the next middleware or route handler
    next();
  } catch (error) {
    console.error("Token verification error:", error);

    // Handle specific JWT errors
    if (error.name === "JsonWebTokenError") {
      return res.status(401).json({ message: "Unauthorized. Invalid token." });
    } else if (error.name === "TokenExpiredError") {
      return res
        .status(401)
        .json({ message: "Unauthorized. Token has expired." });
    } else {
      return res.status(500).json({ message: "Internal server error" });
    }
  }
};

module.exports = authMiddleware;
