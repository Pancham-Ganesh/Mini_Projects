const User = require("../models/user-model")

//? Home page

const home = async (req, res) => {
    try {
        res
        .status(200)
        .json({ msg: "Welcome to controlled HOME page" })

    } catch (error) {
        console.log(error);
        
    }
}

//?User Registration 

const register = async (req, res) => {
    try {

        const { username, email, password, phone, description, occupation, supply } = req.body;

        const userExist = await User.findOne({ email })

        if (userExist) {
            return res.status(200).json({ message : "User already exit"})
        }

        const userCreated = await User.create({ username, email, password, phone, description, occupation, supply});

        res
            .status(201)
            .json({
                msg: "Welcome to controlled register page",
                userId: userCreated._id.toString(),
             });

    } catch (error) {
        res.status(500).json({message: "internal server error"});
        
    }
}

//? User Login

const login = async (req, res) => {
    try {
        
        const { email, password } = req.body;
 
        const userExist = await User.findOne({ email });

        if (!userExist) {
            return res.status(500).json({ message: "Invalid User" })
        }

        const user = await userExist.comparePassword(password);

        if (user) {
             res
            .status(200)
            .json({
                msg: "Login Successful",
                token: await userExist.generateToken(),
                userId: userExist._id.toString(),
                username: userExist.username,
            });
            
        }
        else {
            res.status(401).json({ message: "Invalid email or password"})
        }
        
    } catch (error) {
    res.status(500).json({message:"internal server error"});
    }
}

// to send user-dat : User Logic
const user = async(req, res) => {
    try {
        
        const userData = req.user;
       
        return res.status(200).json({userData})
        
    } catch (error) {
        console.log(`error from the user server ${error}`);
        next(error)
    }
}

    

module.exports = { home, register, login, user };