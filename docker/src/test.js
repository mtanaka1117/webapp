var {PythonShell} = require('python-shell');
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


var options = {
  pythonPath: '/usr/local/bin/python3',
  pythonOptions: ['-u'], 
  // args:['./python/example.csv', './python/result.csv']
};


// PythonShell.run('./python/script.py', options)
//   .then((response) => {
//     console.log(response);
//   })
//   .catch((error) => {
//     console.log(error);
// });

// const { spawn } = require('child_process');
// const pyProg = spawn('python', ['./python/yolov8/install_package.py',"opencv-python"]);

// pyProg.stdout.on('data', function(data) {
//   console.log(data.toString());  
// });
// pyProg.stderr.on('data', (data) => {
//   console.log('Error-->'+ data);
// });


const { exec } = require('child_process');


// PythonShell.run('./yolov8/gpu.py', options)
//   .then((response) => {
//     console.log(response);
//   })
//   .catch((error) => {
//     console.log(error);
// });

const port = 3000
const mysql = require('mysql');

const con = mysql.createConnection({
  host: 'node_docker_db',
  user: 'root',
  password: 'root',
  database:'thermal'
});


// const fs = require('fs');
// const {parse} = require('csv-parse/sync');

// const data = fs.readFileSync('./python/result.csv')
// var res = parse(data);


