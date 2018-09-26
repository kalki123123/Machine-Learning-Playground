# staticMap
Usage of static google maps on smaller screens which have comapratively less resources

Steps :
  1.install node.js,
  
  install and start mongoDB (Or change server.js file accordingly)
  
  2.node server.js                    --- server initial setup
  
  3.npm install ejs --save
  
  4 npm install express --save
  
  5.npm install body-parser --save
  (Install any other dependancy if needed be )
  
  6.node app.js                       --- webserver for this app 
  
  7.point your browser to http://127.0.0.1:8080/trip
  
  8. There is redirection as well for smaller screens with screenwidth less than 600. Change it accordingly to create 
  a redirection to http://127.0.0.1:8080/m.trip
  
  
  Static google map api is used to optimize google maps on smaller screens. It is received as a image and manually new image is created every time user does any of zoom, swipe right/left up/down etc. 
  
  In this particular application, You are able to search a place of your choice,  swipe in all directions using buttons and in build mobile swipes, and Double tap to zoom. 
  
