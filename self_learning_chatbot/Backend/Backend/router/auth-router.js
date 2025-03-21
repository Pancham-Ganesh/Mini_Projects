const express = require("express");
const router = express.Router()
const authcontroller = require("../controllers/auth-controller.js")

// const signupSchema = require("../validators/auth-validator.js")
// const validate = require("../middlewares/validate-middleware.js")
const authMiddleware = require("../middlewares/authMiddleware.js")

router.route("/").get(authcontroller.home)

router.route("/register").post(authcontroller.register)
router.route("/login").post(authcontroller.login)

router.route("/user").get(authMiddleware, authcontroller.user);

module.exports = router; 