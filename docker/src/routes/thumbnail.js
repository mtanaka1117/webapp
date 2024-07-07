var express = require('express');
var router = express.Router();
const mysql = require('mysql');

var log4js = require('log4js');
var accessLogger = log4js.getLogger();

log4js.configure('log-config.json');
router.use(log4js.connectLogger(accessLogger, {level: 'auto'}));

// POSTメッセージの内容をログに記録する関数
function logPostMessage(req, res, next) {
    accessLogger.info(`Received POST request with body: ${JSON.stringify(req.body)}`);
    next();
    }

// ミドルウェアとしてPOSTメッセージの内容を記録
router.use(express.json());
router.use(logPostMessage);

const pool = mysql.createPool({
    host: 'mysql',
    user: 'root',
    password: 'root',
    database: 'thermal',
    dateStrings: 'datetime'
});

router.get('/', function(req, res, next) {
    pool.query('select * from csv', function(err, results) {
        if (err) throw err;
        res.render('thumbnail', {results});
    });
});

router.get('/images/:id', function(req, res, next){
    const image_id = req.params.id;
    pool.query('select * from csv', function(err, results) {
        if (err) throw err;
        detail_Info = results[image_id-1];
        res.render('detail', {detail_Info});
    });
});


router.post('/', function(req, res, next){
    let label = req.body.label;
    let first_time = req.body.date + ' ' + req.body.time;
    let last_time = req.body.date2 + ' ' + req.body.time2;
    let query;

    console.log(first_time, last_time, label);

    if (first_time == ' ' && last_time == ' ' && label == ''){
        query = 'select * from csv'
    }else if (first_time != ' ' && last_time != ' ' && label != ''){
        query = `select * from csv where first_time >= '${first_time}' and last_time <= '${last_time}' and label = '${label}'`;
    }else if (label == ''){
        if (first_time != ' ' && last_time != ' ') query = `select * from csv where first_time >= '${first_time}' and last_time <= '${last_time}'`;
        else if (first_time != ' ') query = `select * from csv where first_time >= '${first_time}'`;
        else query = `select * from csv where last_time <= '${last_time}'`;
    }else{
        if (first_time != ' ') query = `select * from csv where first_time >= '${first_time}' and label = '${label}'`;
        else if (last_time != ' ') query = `select * from csv where last_time <= '${last_time}' and label = '${label}'`;
        else query = `select * from csv where label = '${label}'`;
    }

    pool.query(query, function(err, results){
        if (err) throw err;
        res.render('thumbnail', {results});
    });
});


module.exports = router;