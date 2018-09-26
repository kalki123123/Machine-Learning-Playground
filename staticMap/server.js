//This file is for initial setup - creates DB, table and data
// This should be run once before starting my application
// This application uses mongodb, node.js

var mongo = require('mongodb');

//create database "tripdb" to store location info
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/tripdb";

//Create database
MongoClient.connect(url, function(err, db) {
 if (err) throw err;
 console.log("Database created!");
 db.close();
});

//create colection named 'location' to store location
MongoClient.connect(url, function(err, db) {
 if (err) throw err;
 db.createCollection("location", function(err, res) {
   if (err) throw err;
   console.log("Table created!");
   db.close();
 });
});

//create first object
MongoClient.connect(url, function(err, db) {
 if (err) throw err;
 var myobj = {"lat":49.240157, "lon":6.996933};
 db.collection("location").insertOne(myobj, function(err, res) {
   if (err) throw err;
   console.log("1 record inserted");
   db.close();
 });
});

