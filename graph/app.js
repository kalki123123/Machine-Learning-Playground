// This file is point of entry for my application

// To run application -  node app.js
// point your browser to - http://127.0.0.1:8080/

var fs = require('fs');
var ejs = require('ejs');
var express = require('express');
var app=express();
var port=process.env.PORTH || 8080 ;
var p='hii';
var bodyParser = require('body-parser');
app.use(bodyParser.json());
var mysql = require('mysql');
//DB connection info
var con = mysql.createConnection({
  host: "localhost",
  user: "admin",
  password: "password"

});

//Point of entry in the browser - http://127.0.0.1:8080/
// this loads index.html
app.get('/', function(req, res){
    fs.readFile('./public/index.html', function(err, data) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(data);
    res.end();
  });
});

//To update database when user makes changes in the website
//all records are updated in table - customers

app.post('/update', function (req, res) {


          for(var name in req.body){
          var a=req.body[name].value;
          var b=req.body[name].name;
          var sql = "UPDATE uptain.customers SET value = ? WHERE name = ?";
              con.query(sql, [a,b], function (err, result) {
                if (err) throw err;
                console.log(result.affectedRows + " record(s) updated");
              });
           }



});

//To load values into graph on startup. Values are queried from DB and sent to angularjs
app.get('/get', function (req, res) {
    con.connect(function(err) {

      con.query("SELECT * FROM uptain.customers", function (err, result) {
                if (err) throw err;
                res.send(result);
                });
   });

});

//Listening port - default 8080

app.listen(port , function(){
console.log('listening' + port);
});





