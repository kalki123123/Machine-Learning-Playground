//This file is for initial setup - creates DB, table and data
// This should be run once before starting my application
// This application uses mysql, node.js, angularjs, chartjs

var mysql = require('mysql');

var con = mysql.createConnection({
  host: "localhost",
  user: "admin",
  password: "password"
});
//Create database named - uptain
con.connect(function(err) {
  if (err) throw err;
  console.log("Connected!");
  con.query("CREATE DATABASE IF NOT EXISTS uptain", function (err, result) {
    if (err) throw err;
    console.log("Database created");
  });
// create table customers
var sql = "CREATE TABLE IF NOT EXISTS uptain.customers (name VARCHAR(255), value int)";
  con.query(sql, function (err, result) {
    if (err) throw err;
    console.log("Table created");
  });
//load data into cusotomers. This will be updated from front end later on.
  var sql1 = "INSERT INTO uptain.customers (name, value) VALUES ?";
  var values = [
    ['Jack', 90],
    ['Krish', 52],
    ['Mia', 73],
    ['Anna', 78],
    ['Sophie', 63]
  ];
   con.query(sql1, [values], function(err) {
    if (err) throw err;
    con.end();
    });

});

