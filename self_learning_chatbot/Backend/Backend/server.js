require('dotenv').config();
const cors = require("cors")
const express = require("express");
const app = express();
const authRoute = require("./router/auth-router")
const postRoutes = require('./router/post-router'); // Importing the route we'll create
const connectDb = require("./utils/db");

app.use(express.json());
app.use(cors({
    origin: "*",
    methods: ['GET', 'POST', 'PUT', 'DELETE'], // Specify allowed methods
    credentials: true, // Allow credentials (if needed)
}));

app.use("/api/auth", authRoute);
app.use('/api/post', postRoutes); // API routes for posts


const PORT = 5000;

connectDb().then(() => {
    
    app.listen(PORT, () => {
        console.log(`server is running at port: ${PORT}`);
    })

})