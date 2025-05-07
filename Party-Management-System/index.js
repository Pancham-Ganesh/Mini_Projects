const express = require('express');
const mysql = require("mysql2");
const path = require('path');
const bodyParser = require('body-parser');
let ids=0;
let name="";


const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

const port = 8080;
app.use(express.static('public'));

const connection = mysql.createConnection({
    host: '127.0.0.1',
    port: 3306,
    user: 'root',
    password: 'root',
    database: 'dbproject' 
});

connection.connect(function (err) {
    if (err) {
        console.error('Error connecting to the MySQL server: ' + err.stack);
        return;
    }

    console.log('Connected to the MySQL server.');
});

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});


//class to store information of the user
class Personinfo {
    constructor(id, name) {
        this.id = id;
        this.name = name;
    }
}

// object of the class 
const personInfoInstance = new Personinfo(0, "");

  
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'mainmenu.html'));
});

app.post('/adminlogin', (req, res) => {
    const { username, password } = req.body;

    const selectQuery = `
        SELECT * FROM admin
        WHERE UserName = ? AND Password = ?
    `;

    connection.query(selectQuery, [username, password], (err, results) => {
        if (err) {
            console.error('Error during login: ' + err.message);
            res.status(500).json({ success: false, message: 'Error during login' });
        } else {
            if (results.length > 0) {
                console.log('Admin Login successful for user:', results[0].username);
                res.json({ success: true, message: 'Login successful', username: results[0].username });
            } else {
                console.log('Invalid login credentials');
                res.status(401).json({ success: false, message: 'Invalid login credentials' });
            }
        }
    });
});



app.post('/studentlogin', (req, res) => {
    const { username, password } = req.body;
    const selectQuery = `
        SELECT * FROM students
        WHERE user_name = ? AND Password = ?
    `;

    connection.query(selectQuery, [username, password], (err, results) => {
        if (err) {
            console.error('Error during login: ' + err.message);
            res.status(500).send('Error during login');
        } else {
            console.log('Query Results:', results); 
            if (results.length > 0) {
                const studentData = new Personinfo(results[0].id, username);

                ids = results[0].id;
                name = username;
                console.log('Student Login successful for user:', studentData.name);
                console.log('Student ID:', studentData.id);
                console.log('Student Username:', studentData.name);

                res.json({ success: true, username: studentData.name });
            } else {
                console.log('Invalid login credentials');
                res.status(401).json({ success: false, message: 'Invalid login credentials' });
            }
        }
    });
});



app.post('/studentregister', (req, res) => {
    const { first_name, last_name, user_name, email, password, dietary_preferences, family_members } = req.body;

    const insertQuery = `
        INSERT INTO students (first_name, last_name, user_name, email, password, dietary_preferences, family_members)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    `;

    connection.query(insertQuery, [first_name, last_name, user_name, email, password, dietary_preferences, family_members], (err, results) => {
        if (err) {
            console.error('Error signing up the user: ' + err.message);
            res.status(500).send('Error signing up the user');
        } else {
            console.log('User signed up successfully. ID:', results.insertId);
            res.redirect('/mainmenu.html');
        }
    });
});

app.post('/teacherlogin', (req, res) => {
    const { username, password } = req.body;

    const selectQuery = `
        SELECT * FROM teachers
        WHERE first_name = ? AND Password = ?
    `;

    connection.query(selectQuery, [username, password], (err, results) => {
        if (err) {
            console.error('Error during login: ' + err.message);
            res.status(500).send('Error during login');
        } else {
            if (results.length > 0) {
                console.log('Teacher Login successful for user:', results[0].username);
                res.redirect('/teachersection.html');
            } else {
                console.log('Invalid login credentials');
                res.status(401).send('Invalid login credentials');
            }
        }
    });
});



app.post('/teacherregisteration', (req, res) => {
    const { first_name, last_name, email, password, family_members } = req.body;

    const complimentary = req.body.complimentary === 'on';

    const insertQuery = `
        INSERT INTO teachers (first_name, last_name, email, password, complimentary, family_members)
        VALUES ( ?, ?, ?, ?, ?, ?)
    `;

    connection.query(insertQuery, [first_name, last_name, email, password, complimentary, family_members], (err, results) => {
        if (err) {
            console.error('Error signing up the teacher: ' + err.message);
            res.status(500).send('Error signing up the teacher');
        } else {
            console.log('Teacher signed up successfully. ID:', results.insertId);
            res.redirect('/mainmenu.html');
        }
    });
});

app.post('/submitSuggestion', (req, res) => {
    const { suggestion, budget } = req.body;

    const studentId = ids; 

    const insertQuery = `
        INSERT INTO menu (suggestion_text, student_id, budget)
        VALUES (?, ?, ?)
    `;

    connection.query(insertQuery, [suggestion, studentId, budget], (err, results) => {
        if (err) {
            console.error('Error submitting suggestion: ' + err.message);
            res.status(500).send('Error submitting suggestion');
        } else {
            console.log('Suggestion submitted successfully. ID:', results.insertId);
            res.send('Suggestion submitted successfully');
        }
    });
});


app.get('/suggestedMenus', (req, res) => {
    const selectQuery = `
        SELECT id, suggestion_text
        FROM menu
    `;

    connection.query(selectQuery, (err, results) => {
        if (err) {
            console.error('Error fetching suggested menus: ' + err.message);
            res.status(500).json({ error: 'Failed to fetch suggested menus' });
        } else {
            res.json(results);
        }
    });
});

app.post('/vote', (req, res) => {
    const { menuId } = req.body;
    const studentId = ids; 

    const checkVoteQuery = `
        SELECT vote
        FROM students
        WHERE id = ?
    `;

    connection.query(checkVoteQuery, [studentId], (err, voteResults) => {
        if (err) {
            console.error('Error checking vote status: ' + err.message);
            res.status(500).json({ success: false, error: 'Failed to check vote status' });
        } else {
            const hasVoted = voteResults[0].vote;

            if (!hasVoted) {
                const updateQuery = `
                    UPDATE menu
                    SET votes = votes + 1  -- Change 'votes' to your actual column name representing the number of votes
                    WHERE id = ?
                `;

                connection.query(updateQuery, [menuId], (updateErr, updateResults) => {
                    if (updateErr) {
                        console.error('Error submitting vote: ' + updateErr.message);
                        res.status(500).json({ success: false, error: 'Failed to submit vote' });
                    } else {
                        const setVotedQuery = `
                            UPDATE students
                            SET vote = 1  
                            WHERE id = ?
                        `;

                        connection.query(setVotedQuery, [studentId], (setVotedErr, setVotedResults) => {
                            if (setVotedErr) {
                                console.error('Error updating vote status: ' + setVotedErr.message);
                                res.status(500).json({ success: false, error: 'Failed to update vote status' });
                            } else {
                                res.json({ success: true });
                            }
                        });
                    }
                });
            } else {
                res.json({ success: false, error: 'You have already voted' });
            }
        }
    });
});

app.post('/submitPerformance', (req, res) => {
    const { performanceType, duration, specialRequirements, budget, Themes } = req.body;

    const studentId = ids; 

    const insertPerformanceQuery = `
      INSERT INTO suggestions (studentid, performancetype, duration, specialrequirements, budget)
      VALUES (?, ?, ?, ?, ?)
    `;

    connection.query(
        insertPerformanceQuery,
        [studentId, performanceType, duration, specialRequirements, budget],
        (err, results) => {
            if (err) {
                console.error('Error submitting performance: ' + err.message);
                res.status(500).send('Error submitting performance');
            } else {
                const performanceId = results.insertId;

                const insertThemesQuery = `
                  INSERT INTO themes (student_id, name)
                  VALUES (?, ?)
                `;

                connection.query(
                    insertThemesQuery,
                    [studentId, Themes],
                    (errThemes) => {
                        if (errThemes) {
                            console.error('Error submitting theme: ' + errThemes.message);
                            res.status(500).send('Error submitting theme');
                        } else {
                            console.log('Performance and theme submitted successfully. Performance ID:', performanceId);
                            res.send('Performance and theme submitted successfully');
                        }
                    }
                );
            }
        }
    );
});

app.get('/suggestedPerformances', (req, res) => {
    const selectQuery = `
        SELECT id, performancetype, votes
        FROM suggestions
    `;

    connection.query(selectQuery, (err, results) => {
        if (err) {
            console.error('Error fetching suggested performances: ' + err.message);
            res.status(500).json({ error: 'Failed to fetch suggested performances' });
        } else {
            res.json(results);
        }
    });
});

app.post('/p_vote', (req, res) => {
    const { performanceId } = req.body;

    const updateQuery = `
        UPDATE suggestions
        SET votes = votes + 1  -- Change 'votes' to your actual column name representing the number of votes
        WHERE id = ?
    `;

    connection.query(updateQuery, [performanceId], (updateErr, updateResults) => {
        if (updateErr) {
            console.error('Error submitting vote: ' + updateErr.message);
            res.status(500).json({ success: false, error: 'Failed to submit vote' });
        } else {
            res.json({ success: true });
        }
    });
});

app.get('/attendance', (req, res) => {
    const queryStudents = `SELECT count(*) AS studentCount FROM students`;
    const queryTeachers = `SELECT count(*) AS teacherCount FROM teachers`;
    const queryTeacherFamilies = `SELECT sum(family_members) AS teacherFamilyCount FROM teachers`;
    const queryStudentFamilies = `SELECT sum(family_members) AS studentFamilyCount FROM students`;
  
    connection.query(queryStudents, (errStudents, resultsStudents) => {
      if (errStudents) {
        console.error('Error querying students: ' + errStudents.message);
        res.status(500).json({ error: 'Error querying students' });
        return;
      }
  
      connection.query(queryTeachers, (errTeachers, resultsTeachers) => {
        if (errTeachers) {
          console.error('Error querying teachers: ' + errTeachers.message);
          res.status(500).json({ error: 'Error querying teachers' });
          return;
        }
  
        connection.query(queryTeacherFamilies, (errTeacherFamilies, resultsTeacherFamilies) => {
          if (errTeacherFamilies) {
            console.error('Error querying teacher families: ' + errTeacherFamilies.message);
            res.status(500).json({ error: 'Error querying teacher families' });
            return;
          }
  
          connection.query(queryStudentFamilies, (errStudentFamilies, resultsStudentFamilies) => {
            if (errStudentFamilies) {
              console.error('Error querying student families: ' + errStudentFamilies.message);
              res.status(500).json({ error: 'Error querying student families' });
              return;
            }
  
            const counts = {
              studentCount: resultsStudents[0].studentCount,
              teacherCount: resultsTeachers[0].teacherCount,
              teacherFamilyCount: resultsTeacherFamilies[0].teacherFamilyCount,
              studentFamilyCount: resultsStudentFamilies[0].studentFamilyCount,
            };
  
            res.json(counts);
          });
        });
      });
    });
  });

app.get('/themesdisplay', (req, res) => {
    const selectThemesQuery = `
        SELECT name
        FROM themes
    `;

    connection.query(selectThemesQuery, (err, results) => {
        if (err) {
            console.error('Error fetching theme names: ' + err.message);
            res.status(500).json({ error: 'Failed to fetch theme names' });
        } else {
            const themeNames = results.map(result => result.name);
            res.json(themeNames);
        }
    });
});

app.get('/highestVotedSuggestion', (req, res) => {
    const selectHighestVotedQuery = `
        SELECT suggestion_text
        FROM menu
        ORDER BY votes DESC
        LIMIT 1
    `;

    connection.query(selectHighestVotedQuery, (err, results) => {
        if (err) {
            console.error('Error fetching highest voted suggestion: ' + err.message);
            res.status(500).json({ error: 'Failed to fetch highest voted suggestion' });
        } else {
            if (results.length > 0) {
                const highestVotedSuggestion = results[0].suggestion_text;
                res.json({ highestVotedSuggestion });
            } else {
                res.json({ highestVotedSuggestion: 'No suggestions available' });
            }
        }
    });
});

app.get('/highestVotedPerformance', (req, res) => {
    const selectHighestVotedPerformanceQuery = `
        SELECT performancetype AS highestVotedPerformanceType
        FROM suggestions
        ORDER BY votes DESC
        LIMIT 1
    `;

    connection.query(selectHighestVotedPerformanceQuery, (err, results) => {
        if (err) {
            console.error('Error fetching highest voted performance type: ' + err.message);
            res.status(500).json({ error: 'Failed to fetch highest voted performance type' });
        } else {
            if (results.length > 0) {
                const highestVotedPerformanceType = results[0].highestVotedPerformanceType;
                res.json({ highestVotedPerformanceType });
            } else {
                res.json({ highestVotedPerformanceType: 'No performances available' });
            }
        }
    });
});
