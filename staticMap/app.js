// This file is point of entry for my application

// To run application -  node app.js
// point your browser to - http://127.0.0.1:8080/trip

var fs = require('fs');
var ejs = require('ejs');
var express = require('express');
var app=express();
var port=process.env.PORTH || 8080 ;
var p='hii';
var bodyParser = require('body-parser');
app.use(bodyParser.json());


var http = require('http');
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/tripdb";





//Point of entry in the browser - http://127.0.0.1:8080/trip
// this loads index.html
app.get('/trip', function(req, res){
    fs.readFile('./public/index.html', function(err, data) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(data);
    res.end();
  });
});
// Point of redirection for smaller screens http://127.0.0.1:8080/m.trip
app.get('/m.trip', function(req, res){
    fs.readFile('./public/m_index.html', function(err, data) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(data);
    res.end();
  });
});


//To load values into graph on startup. Values are queried from DB and sent to angularjs
app.get('/get', function (req, res) {
   MongoClient.connect(url, function(err, db) {
  if (err) throw err;
  db.collection("location").findOne({}, function(err, result) {
    if (err) throw err;
    console.log(result.lat);
    console.log(result.lon);
    res.send(result);
    db.close();
  });
});


});

//Listening port - default 8080

app.listen(port , function(){
console.log('listening' + port);
});





