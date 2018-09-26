# data challenge
This is Luminovo's data challenge.

The dataset you are dealing with is a list of features extracted from human blood cells. Each line in `features.csv` corresponds to a different patient. `labels.csv` contains matching labels (0 = healthy, 1 = sick). For the purpose of this challenge, we don't expect any medical knowledge or insights (we don't have any either) and therefore removed the names of all the features, so don't worry about where they came from, but do worry about how to use them best.

Your task is to build an algorithm that will help us identify whether a patient's cells are healthy or sick, just in case one of us falls sick in the office. We probably care more about identifying all the sick people than making a mistake about a healthy person every now and then.

Unfortunately we don't have an extra test set, so you will have to use the data available here to train your model and to tell us how well your algorithm will work for new patients. 

When you get back to us with your code and your results, we expect you to walk us through what you did. We not only care about the best result you got, but about how you got there.

Please prepare a `shiny-results.ipynb` notebook that we will use as the basis for our discussion. It can contain interesting visualizations or error analysis you did to help you guide your model choices. It should probably contain a proper evaluation of your final model. If something didn't work, but you think it is interesting to talk about why, you can put it in here too.

Last but not least, please use Python 3 for this challenge because, you know, you might survive the next Mayan apocalypse but Python 2 won't.

## recommended (but not required) folder structure
similar to the one we use at internal projects.

```
├── README.md          <- the top-level README you are reading right now.
│
├── requirements.txt   <- please put the required dependencies here 
│	            	  so that we can run your code if we want to
│
├── data	       
│   ├── features.csv   <- the black-box features
│   └── labels.csv     <- binary labels
│
├── log                <- put your checkpoints of trained models, evaluations 
│			  and other logs here
│
├── notebooks          <- put your Jupyter notebooks here.
│
└── src                <- put your source code and scripts here.
```
