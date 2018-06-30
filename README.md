
# Recsys Challenge 2018 - Main Track

Team: cocoplaya

Members: 

 - Andres Ferraro
 - Dmitry Bogdanov
 - Jason Yoon
 - Lucas Kim

Username in recsys-challenge.spotify.com: aferraro

Instructions
-----------

In order to reproduce the recommendations submitted for the Main track of the challenge the model_main.py file must be executed.

First we train a Matrix Factorization model and then we generate the recommendations based on this model to the playlists in the challenge_set, then we combine this results with other model which computes the most probable songs based on the coocurrence in the playlists. This is all computed in the file model_main.py.

To compute the Matrix Factorization we use the library LightFM, in requirements.txt you can find all the dependencies, we use the version 3.6.4 of Python.
Inside model_main.py you can specify where the MPD dataset is located and were the challenge_set is located. Also in this file you can specify where the output CSV should be placed.

