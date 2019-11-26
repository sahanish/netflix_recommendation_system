# Netflix Recommendation System

=====================Introduction========================

The Recommender System Web Application serves personalized recommendations of movies from a database, based on the logged 
in user's past ratings and by that, his/her similarity with other users in the database. The application also serves 
recommendations of items similar to the item recently rated by that user.

=====================Algorithms used========================

The Web Application is served using the Flask framework. There are primarily two algorithms used to generate recommendations:
1. Matrix Factorization in Keras 2 - For prediction of missing ratings in the input ratings matrix
2. Collaborative Filtering - For recommending items similar to the item recently rated by a user

=====================Project Directory Structure========================

1. __init__.py
For importing Flask variables and creating a Database instance

2. models.py
For defining classes of the four tables used in this project:
a)Users - Records of all users in the database
b)Movies - Records of all movies in the database
c)Interactions - Records of all user ratings used for training the Matrix Factorization Algorithm
d)Validations - Records of all user ratings used for validating the Matrix Factorization Algorithm

3. forms.py
For defining classes of the three types of forms used in this project:
a)Registration form - For adding new users
b)Login form - For logging in a user
c)Search form - For searching a movie in the website

4. routes.py
The core of this project, where all the route functions are placed. The various routes are - 

a) /index - The homepage, with links to all other pages, can only be accessed by logged in users

b) /login - The login page, for logging into the recommender website

c) /home - The main page, where the saved Machine Learning model is loaded upon each user's login, and personalized 
            movie recommendations displayed
            
d) /search/<data> -  The page that returns the movie search result

e) /logout - When a user clicks the logout link, controls returns to the login page

f) /register - The new user registration page

g) /received_rating - When a user rates an item, the information is received here using request to a jquery, after which the
                      saved Machine Learning model is retrained with the new record
                      
h) /refresh_recommendations - Once the Machine Learning model is retrained, clicking on a Refresh Recommendations link on
                              the recommendations page reloads the page with the updated recommendations
                              
5. index.html - The index page

6. login.html - The login page

7. home.html - The recommendations page

8. register.html - The registration page

9. search.html - The movie search page

=====================Database========================

For this project, i have used the open source SQLite3 database. 


=====================Recommender Performance========================

The Matrix Factorization algorithm used for predicting missing ratings is tested using Mean Squared Error(MSE) accuracy metric. It converges successfully with the following results:
MSE(Training Data) :  0.51
MSE(Testing Data) : 0.56

======================Data========================

The Matrix Factorization algorithm is trained on 5188 samples, taken from 50 users and 200 movies. The corresponding csv file, edited_ratings_subset_larger.csv, has been included in my project repo.
