const mysql = require('mysql');
const conn = mysql.createConnection({
  host: 'mysql',
  user: 'root',
  password: 'root',
  database: 'thermal'
});

conn.connect();

conn.query('SELECT * FROM csv', function (error, results, fields) {
  if (error) throw error;
  console.log(results);
});

conn.end();