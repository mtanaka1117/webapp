var express = require('express');
var router = express.Router();

const mysql = require('mysql');
const conn = mysql.createConnection({
  host: 'mysql',
  user: 'root',
  password: 'root',
  database: 'thermal',
  dateStrings: 'datetime'
});

router.get('/', function(req, res, next) {
  conn.connect();
  conn.query('SELECT * FROM csv', function(error, results) {
    if (error) throw error;
    res.render('db', {results});
  });
  conn.end();
});
  
module.exports = router;