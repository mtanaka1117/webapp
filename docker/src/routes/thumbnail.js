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
        res.render('thumbnail', {results});
    });
    conn.end();
});

router.get('/detail/:id', function(req, res, next){
    const image_id = req.params.id
    conn.query('SELECT * FROM csv', function(error, results) {
        detail_Info = results[image_id-1]
        res.render('detail', {detail_Info});
    });
});
  
module.exports = router;