## Exercise 2 Overview


Given the dataset that contains

| Composition title | Composition Writers | Recording Title | Recording Writes |Action |
| --- | --- |--- | --- |--- |
| Yellow submarine | Leo Ouha |Yellow submarine(remix) |Leo Ouha |Leo Ouha |ACCEPTED |
| Shape of you | Ed Sheeran| Anaconda | Mick George | Roco Selto |Leo REJECTED |

Train an ML/DL model for pair matching of compositions and recordings.
Report and evaluate the results.


For this problem as you can see in the exercise_2 notebook i followed this procedure:

- Retrieve and clean dataset.
- Vectorize it using tf-idf vectorizer.
- Reduce the dimentionality of the input feature vectors with SVD.
- Run multiple ML algorithms with increased complexity.
- Run a final CNN that performs convolutions with multiple filters in order to find the most important quantities of the input.

