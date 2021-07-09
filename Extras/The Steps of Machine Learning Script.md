# The Steps of Machine Learning Script

1. Load in the data
	- X and Y
	- Typically use Pandas unless the data is too complex

2. Split data into train/test sets
	- Sometimes "test" and "validation" are used interchangeably, and the "true test set" is sometimes else .
3. Build a model
	- OOP 
	- Scikit-Learn
	- Tensorflow 2.0 standard is keras API 
4. Fit the model (gradient descent)
	- model.fit(X,Y)
5. Evaluate the model
6. Make predictions
	- model.predict(X)