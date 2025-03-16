const express = require("express");
const router = express.Router()
const authcontroller = require("../controllers/auth-controller.js")
const upload = require("../middlewares/upload.js"); // ES Modules
const uploadFile = require("../cloud/s3bucket.js"); // Import upload controller

// const signupSchema = require("../validators/auth-validator.js")
// const validate = require("../middlewares/validate-middleware.js")
const authMiddleware = require("../middlewares/authMiddleware.js")

router.route("/").get(authcontroller.home)

router.route("/register").post(authcontroller.register)
router.route("/login").post(authcontroller.login)

router.route("/user").get(authMiddleware, authcontroller.user);

router.post("/upload", authMiddleware, upload.single("image"), uploadFile);

module.exports = router; 