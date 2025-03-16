const jwt = require("jsonwebtoken")
const User = require("../models/user-model")

// user data by verification without password
const authMiddleware = async(req, res, next) => {
    const token = req.header('Authorization')

    if (!token) {
        res.status(401).json("Unauthorized HTTP, Token Not Provided")
    }

    const jwtToken = token.replace('Bearer', "").trim(); 
    try {
        const isVerified = jwt.verify(jwtToken, process.env.JWT_SECRET_KEY)
        const userData = await User.findOne({ email: isVerified.email }).select({ password: 0 })
        
        req.user = userData
        req.token = token
        req.userId = userData._id
    } catch (error) {
        console.log(error)
        return res.status(401).json({ msg: "Unauthorized. Invalid token." })
    }   
    next();
}

module.exports = authMiddleware;