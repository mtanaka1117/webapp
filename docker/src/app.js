var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;

const port = 3000
const mysql = require('mysql');

const con = mysql.createConnection({
  host: 'node_docker_db',
  user: 'root',
  password: 'root',
  database:'thermal'
});




// const sql = "INSERT INTO csv(label,first_time,last_time,count,bbox) VALUES('76.0','2023-12-01 11:18:10.137639','2023-12-01 11:18:10.464388','5','[439.6739, 0.2555, 631.5266, 208.188]')"

// con.query(sql,function(err, result, fields){
// 	if (err) throw err;
// 	console.log(result)
// })




