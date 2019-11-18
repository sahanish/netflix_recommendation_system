from flask import render_template,url_for,flash,redirect, request, jsonify, session
from sqlalchemy import func
from werkzeug.urls import url_parse
from flask_login import current_user, login_user, logout_user, login_required
from app import app, db
from app.forms import LoginForm, RegistrationForm, SearchForm   
from app.models import Users, Movies, Interactions, Validations
from app.templates.csv_migrations import migrate_interactions,migrate_customers,migrate_movies
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import keras
from keras.layers import Input, Embedding, Dot, Add, Flatten, Concatenate, Dense
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.models import load_model
np.random.seed(1337)
from keras.models import load_model, Model
from keras import backend as Kb
from sqlalchemy import desc
import os
import json
@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods = ['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():#Seperate users requesting login    form and users submitting form
        user = Users.query.filter_by(username = form.username.data).first()
        if user is None or form.password.data!=user.password:
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user,remember = form.remember_me.data)
        next_page = request.args.get('next')
        #next_page = url_for('home')
        if not next_page or url_parse(next_page).netloc !='':#Risky domain
            next_page = url_for('home')
        return redirect(next_page)
    return render_template('login.html', title = 'User Login', form = form)#Only if user has requested login page

def IICF_common_ratings(a,b,df1):
    movie_a = df1[df1.userid.isin(np.array(range(50)))][df1.movieid==a]
    movie_b = df1[df1.userid.isin(np.array(range(50)))][df1.movieid==b]
    movie_a_ids = movie_a.userid.values
    movie_b_ids = movie_b.userid.values
    common = set(movie_a_ids) & set(movie_b_ids)
    A = movie_a[movie_a.userid.isin(common)].sort_values(by = 'userid', ascending = True).rating.values
    B = movie_b[movie_b.userid.isin(common)].sort_values(by = 'userid', ascending = True).rating.values
    return [A,B]

def similarity(a,b):
    num = (a - np.mean(a)).dot(b.T - np.mean(b.T))
    den = np.sqrt(((a.T - np.mean(a.T)).dot((a - np.mean(a))))*((b.T - np.mean(b.T)).dot((b - np.mean(b))))
             )
    return num/den

def recommender(user,N, model_name):
    Kb.clear_session()
    model = load_model(os.path.join(os.path.dirname(__file__),model_name))
    users = Interactions.query.with_entities(Interactions.userid).all()
    users = [int(u) for u, in users]
    movies = Interactions.query.with_entities(Interactions.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Interactions.query.with_entities(Interactions.rating).all()
    ratings = [r for r, in ratings]
    ratings_train = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_train = pd.DataFrame(ratings_train)
    # commons = [UUCF_common_ratings(35,i,ratings_train) for i in list(range(50))]
    # all_sims = {
    # 'user':list(range(50)),
    # 'similarity':[similarity(commons[i][0],commons[i][1]) for i in range(50)]
    # }
    # sim_df = pd.DataFrame(all_sims)
    # sim_df = sim_df.sort_values(by = 'similarity',ascending = False)[0:10]
    # sim_users = sim_df.user.values[[range(1,3)]]
    mu = ratings_train.rating.mean()
    movies_train = list(ratings_train[ratings_train.userid==user].movieid.values)
    movies_train = [int(x) for x in movies_train]
    # neighbour1_movies = list(ratings_train[ratings_train.userid==sim_users[0]].movieid.values)
    # neighbour2_movies = list(ratings_train[ratings_train.userid==sim_users[1]].movieid.values)

    all_movies = range(200)
    # unrated_movies = [i for i in all_movies if i not in movies_train if any([i in neighbour1_movies, i in neighbour2_movies])]
    unrated_movies = [i for i in all_movies if i not in movies_train]

    predictions = mu + model.predict(
    x = [np.ones(len(unrated_movies))*user,np.array(unrated_movies)]
    )

    data = {'Movie' : unrated_movies,
            'Prediction':[ratings for [ratings] in predictions]
            }
            
    my_movies = pd.DataFrame(data)
    my_movies = my_movies.sort_values(by = 'Prediction',ascending = False)[0:N]
    ids = [int(x) for x in list(my_movies['Movie'].values)]
    return [Movies.query.filter_by(movieid=id).one() for id in ids]


def popular_recommender(user):

    return Movies.query.order_by(desc(Movies.avg_rating)).filter(Movies.avg_rating>=4)

def matrix_factorization_retrain(user, N, model_name):

    users = Interactions.query.with_entities(Interactions.userid).all()
    users = [int(u) for u, in users]
    movies = Interactions.query.with_entities(Interactions.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Interactions.query.with_entities(Interactions.rating).all()
    ratings = [r for r, in ratings]
    ratings_train = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_train = pd.DataFrame(ratings_train)
    users = Validations.query.with_entities(Validations.userid).all()
    users = [int(u) for u, in users]
    movies = Validations.query.with_entities(Validations.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Validations.query.with_entities(Validations.rating).all()
    ratings = [r for r, in ratings]
    ratings_test = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_test = pd.DataFrame(ratings_test)
    Kb.clear_session()
    K = 10
    mu = ratings_train.rating.mean()
    epochs=50
    reg = 0.001
    u = Input(shape = (1,))
    m = Input(shape = (1,))
    N = 100
    M = 500
    u_embedding = Embedding(N,K,embeddings_regularizer = l2(reg))(u)
    m_embedding = Embedding(M,K,embeddings_regularizer = l2(reg))(m)

    u_bias = Embedding(N,1,embeddings_regularizer = l2(reg))(u)
    m_bias = Embedding(M,1,embeddings_regularizer = l2(reg))(m)
    x = Dot(axes = 2)([u_embedding,m_embedding])

    x = Add()([x, u_bias, m_bias])  
    x = Flatten()(x)

    model = Model(inputs = [u, m], output = x)

    model.compile(
    loss = 'mse',
    # optimizer = SGD(lr = 0.009, momentum = 0.9),
    optimizer = 'adam',
    metrics = ['mse']
    )

    r = model.fit(
    x = [ratings_train.userid.values, ratings_train.movieid.values],
    y = ratings_train.rating.values - mu,
    epochs = epochs,
    batch_size = 64,
    validation_data = (
        [ratings_test.userid.values, ratings_test.movieid.values],
        ratings_test.rating.values - mu
    ),
    shuffle = False
    )

    model.save(os.path.join(os.path.dirname(__file__), model_name))

def ANN_retrain(user, N, model_name):

    users = Interactions.query.with_entities(Interactions.userid).all()
    users = [int(u) for u, in users]
    movies = Interactions.query.with_entities(Interactions.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Interactions.query.with_entities(Interactions.rating).all()
    ratings = [r for r, in ratings]
    ratings_train = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_train = pd.DataFrame(ratings_train)
    users = Validations.query.with_entities(Validations.userid).all()
    users = [int(u) for u, in users]
    movies = Validations.query.with_entities(Validations.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Validations.query.with_entities(Validations.rating).all()
    ratings = [r for r, in ratings]
    ratings_test = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_test = pd.DataFrame(ratings_test)
    Kb.clear_session()
    K = 10
    N = 100
    M = 500
    mu = ratings_train.rating.mean()
    epochs=40
    reg = 0.15
    u = Input(shape = (1,))
    m = Input(shape = (1,))
    u_embedding = Embedding(N,K,embeddings_regularizer = l2(reg))(u)
    m_embedding = Embedding(M,K,embeddings_regularizer = l2(reg))(m)
    u_embedding = Flatten()(u_embedding)#NxK
    m_embedding = Flatten()(m_embedding)#MxK
    x = Concatenate()([u_embedding, m_embedding])#Nx2K

    #MLP
    x = Dense(1000, activation = 'relu')(x)
    x = Dense(1)(x)
    model = Model(inputs = [u, m], output = x)

    model.compile(
    loss = 'mse',
    # optimizer = SGD(lr = 0.009, momentum = 0.9),
    optimizer = 'adam',
    metrics = ['mse']
    )

    r = model.fit(
    x = [ratings_train.userid.values, ratings_train.movieid.values],
    y = ratings_train.rating.values - mu,
    epochs = epochs,
    batch_size = 64,
    validation_data = (
        [ratings_test.userid.values, ratings_test.movieid.values],
        ratings_test.rating.values - mu
    ),
    shuffle = False
    )

    model.save(os.path.join(os.path.dirname(__file__), model_name))


def RESL_retrain(user, N, model_name):

    users = Interactions.query.with_entities(Interactions.userid).all()
    users = [int(u) for u, in users]
    movies = Interactions.query.with_entities(Interactions.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Interactions.query.with_entities(Interactions.rating).all()
    ratings = [r for r, in ratings]
    ratings_train = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_train = pd.DataFrame(ratings_train)
    users = Validations.query.with_entities(Validations.userid).all()
    users = [int(u) for u, in users]
    movies = Validations.query.with_entities(Validations.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Validations.query.with_entities(Validations.rating).all()
    ratings = [r for r, in ratings]
    ratings_test = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_test = pd.DataFrame(ratings_test)
    Kb.clear_session()
    K = 10
    N = 100
    M = 500

    #Matrix Factorization Branch
    mu = ratings_train.rating.mean()
    epochs=40
    reg = 0.15
    u = Input(shape = (1,))
    m = Input(shape = (1,))
    u_embedding = Embedding(N,K,embeddings_regularizer = l2(reg))(u)
    m_embedding = Embedding(M,K,embeddings_regularizer = l2(reg))(m)

    u_bias = Embedding(N,1,embeddings_regularizer = l2(reg))(u)
    m_bias = Embedding(M,1,embeddings_regularizer = l2(reg))(m)
    x = Dot(axes = 2)([u_embedding,m_embedding])

    x = Add()([x, u_bias, m_bias])  
    x = Flatten()(x)

    #ANN Branch

    reg = 0.2
    u_embedding = Embedding(N,K,embeddings_regularizer = l2(reg))(u)
    m_embedding = Embedding(M,K,embeddings_regularizer = l2(reg))(m)
    u_embedding = Flatten()(u_embedding)#NxK
    m_embedding = Flatten()(m_embedding)#MxK
    y = Concatenate()([u_embedding, m_embedding])#Nx2K
    y = Dense(1000, activation = 'relu')(y)
    y = Dense(1)(y)

    #Residual Learning

    x = Add()([x, y])
    model = Model(inputs=[u, m], outputs=x)

    model.compile(
    loss = 'mse',
    optimizer = 'adam',
    metrics = ['mse']
    )

    r = model.fit(
    x = [ratings_train.userid.values, ratings_train.movieid.values],
    y = ratings_train.rating.values - mu,
    epochs = epochs,
    batch_size = 64,
    validation_data = (
        [ratings_test.userid.values, ratings_test.movieid.values],
        ratings_test.rating.values - mu
    ),
    shuffle = False
    )

    model.save(os.path.join(os.path.dirname(__file__), model_name))




def collaborative_filtering(user, movie, N):
    users = Interactions.query.with_entities(Interactions.userid).all()
    users = [int(u) for u, in users]
    movies = Interactions.query.with_entities(Interactions.movieid).all()
    movies = [int(m) for m, in movies]
    ratings = Interactions.query.with_entities(Interactions.rating).all()
    ratings = [r for r, in ratings]
    ratings_train = {'userid':users,'movieid':movies,'rating':ratings}
    ratings_train = pd.DataFrame(ratings_train)
    commons = [IICF_common_ratings(movie,i,ratings_train) for i in list(range(200))]
    all_sims = {
    'movie':list(range(200)),
    'similarity':[similarity(commons[i][0],commons[i][1]) for i in range(200)]
    }
    sim_df = pd.DataFrame(all_sims)
    sim_df = sim_df.sort_values(by = 'similarity',ascending = False)
    sim_movies = sim_df.movie.values[1:6]
    ids = [int(x) for x in list(sim_movies)]
    print("Collaborative filtering generates:", ids)
    return [Movies.query.filter_by(movieid=id).one() for id in ids]

    



@app.route('/home', methods = ['GET', 'POST'])
@login_required
def home():
    user = current_user.userid
    form = SearchForm()
    if form.validate_on_submit():
        print(form.search.data)
        return redirect(url_for('search', data = form.search.data))

    n_watched = Interactions.query.filter_by(userid = user).count()
    print(user, n_watched)
    if n_watched>=10:
        # topN1 = recommender(user,5, 'MF_shf.h5')
        topN1 = recommender(user,5, 'MF_model_shf.h5')
        topN2 = popular_recommender(user)


        recommendations1 = [
            {
            'thumbnail': topN1[0].thumbnail,
            'watchlink': topN1[0].watchlink,
            'avg_rating': round(topN1[0].avg_rating,1),
            'movieid': topN1[0].movieid
            },
            {
            'thumbnail': topN1[1].thumbnail,
            'watchlink': topN1[1].watchlink,
            'avg_rating': round(topN1[1].avg_rating,1),
            'movieid': topN1[1].movieid
            },
            {
            'thumbnail': topN1[2].thumbnail,
            'watchlink': topN1[2].watchlink,
            'avg_rating': round(topN1[2].avg_rating,1),
            'movieid': topN1[2].movieid
            },
            {
            'thumbnail': topN1[3].thumbnail,
            'watchlink': topN1[3].watchlink,
            'avg_rating': round(topN1[3].avg_rating,1),
            'movieid': topN1[3].movieid
            },
            {
            'thumbnail': topN1[4].thumbnail,
            'watchlink': topN1[4].watchlink,
            'avg_rating': round(topN1[4].avg_rating,1),
            'movieid': topN1[4].movieid
            }
        ]

        recommendations2 = [
            {
            'thumbnail': topN2[0].thumbnail,
            'watchlink': topN2[0].watchlink,
            'avg_rating': round(topN2[0].avg_rating,1),
            'movieid': topN2[0].movieid
            },
            {
            'thumbnail': topN2[1].thumbnail,
            'watchlink': topN2[1].watchlink,
            'avg_rating': round(topN2[1].avg_rating,1),
            'movieid': topN2[1].movieid
            },
            {
            'thumbnail': topN2[2].thumbnail,
            'watchlink': topN2[2].watchlink,
            'avg_rating': round(topN2[2].avg_rating,1),
            'movieid': topN2[2].movieid
            },
            {
            'thumbnail': topN2[3].thumbnail,
            'watchlink': topN2[3].watchlink,
            'avg_rating': round(topN2[3].avg_rating,1),
            'movieid': topN2[3].movieid
            },
            {
            'thumbnail': topN2[4].thumbnail,
            'watchlink': topN2[4].watchlink,
            'avg_rating': round(topN2[4].avg_rating,1),
            'movieid': topN2[4].movieid
            }
        ]

        dummy_recommendations = [
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            }
        ]
        if bool(session.get(str(user))):

            which_movie = session[str(user)]['which_movie']
            movie_watched = Movies.query.filter_by(movieid=which_movie).one().moviename
            return render_template('home.html', recommendations1 = recommendations1, recommendations2 = recommendations2, recommendations3 = session[str(user)]['recommendations'], n_watched = n_watched, which_movie = movie_watched, colab = True, form = form)
        else:
            return render_template('home.html', recommendations1 = recommendations1, recommendations2 = recommendations2, recommendations3 = dummy_recommendations, n_watched = n_watched, colab = False, form = form)





    else:
        topN1 = popular_recommender(user)

        recommendations1 = [
            {
            'thumbnail': topN1[0].thumbnail,
            'watchlink': topN1[0].watchlink,
            'avg_rating': round(topN1[0].avg_rating,1),
            'movieid': topN1[0].movieid
            },
            {
            'thumbnail': topN1[1].thumbnail,
            'watchlink': topN1[1].watchlink,
            'avg_rating': round(topN1[1].avg_rating,1),
            'movieid': topN1[1].movieid
            },
            {
            'thumbnail': topN1[2].thumbnail,
            'watchlink': topN1[2].watchlink,
            'avg_rating': round(topN1[2].avg_rating,1),
            'movieid': topN1[2].movieid
            },
            {
            'thumbnail': topN1[3].thumbnail,
            'watchlink': topN1[3].watchlink,
            'avg_rating': round(topN1[3].avg_rating,1),
            'movieid': topN1[3].movieid
            },
            {
            'thumbnail': topN1[4].thumbnail,
            'watchlink': topN1[4].watchlink,
            'avg_rating': round(topN1[4].avg_rating,1),
            'movieid': topN1[4].movieid
            }
        ]

        dummy_recommendations = [
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            },
            {
            'thumbnail': "NA",
            'watchlink': "NA",
            'avg_rating': 0,
            'movieid': "NA"
            }
        ]
        if bool(session.get(str(user))):

            #recommendations3 = session[str(user)]['recommendations']
            which_movie = session[str(user)]['which_movie']
            movie_watched = Movies.query.filter_by(movieid=which_movie).one().moviename
            return render_template('home.html', recommendations1 = recommendations1, recommendations2 = session[str(user)]['recommendations'], recommendations3 = dummy_recommendations, n_watched = n_watched, which_movie = movie_watched, colab = True,form = form)
        else:
            return render_template('home.html', recommendations1 = recommendations1, recommendations2 = dummy_recommendations, recommendations3 = dummy_recommendations, n_watched = n_watched, colab = False,form = form)


@app.route('/search/<data>')
def search(data):
    movie = Movies.query.filter_by(moviename = data).first()
    if movie is not None:
        print(movie.thumbnail, " is found")
        search_result = {
            'thumbnail': movie.thumbnail,
            'watchlink': movie.watchlink,
            'avg_rating': round(movie.avg_rating,1),
            'movieid': movie.movieid
        }
        return render_template('search.html', search_result = search_result, found = 1)
    else:
        return render_template('search.html', search_error = "Movie not found", found = 0) 
        
    




@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods = ['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():#Seperate users requesting registeration form and users submitting form
        user = Users(
            userid = db.session.query(func.count(Users.id)).scalar(),
            username = form.username.data,
            email = form.email.data,
            password = form.password.data
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('registered'))
    return render_template('register.html', title = 'User Registration', form = form)#Only if user has requested register page


@app.route('/user_registered', methods = ['GET', 'POST'])
def registered():
    return render_template('user_registered.html')


@app.route('/received_rating', methods = ['GET', 'POST'])
def received_rating():
    if request.method == "POST":
        data = request.json
        record = Interactions(
            userid = int(data['user']),
            movieid = int(data['movie']),
            rating = data['rating']
        )
        exist_record = Interactions.query.filter_by(userid = record.userid, movieid = record.movieid).first()
        if not bool(exist_record):
            db.session.add(record)
            db.session.commit()
            update_movie = Movies.query.filter_by(movieid = record.movieid).first()
            if bool(update_movie):
                update_movie.avg_rating = ((update_movie.avg_rating*update_movie.N_ratings)+record.rating)/(update_movie.N_ratings+1)
                update_movie.N_ratings+=1
                db.session.commit()
        else:
            update_movie = Movies.query.filter_by(movieid = record.movieid).first()
            if bool(update_movie):
                update_movie.avg_rating = ((update_movie.avg_rating*update_movie.N_ratings)+record.rating-exist_record.rating)/update_movie.N_ratings
            exist_record.rating = record.rating
            db.session.commit()
            
           


        user = current_user.userid
        n_watched = Interactions.query.filter_by(userid = user).count()
        if n_watched>=10:
            # matrix_factorization_retrain(user,5, 'MF_shf.h5')
            matrix_factorization_retrain(user,5, 'MF_model_shf.h5')
        if float(record.rating)>=3.5:
            topN = collaborative_filtering(user,int(record.movieid),5)
            recommendations = [
            {
            'thumbnail': topN[0].thumbnail,
            'watchlink': topN[0].watchlink,
            'avg_rating': round(topN[0].avg_rating,1),
            'movieid': topN[0].movieid
            },
            {
            'thumbnail': topN[1].thumbnail,
            'watchlink': topN[1].watchlink,
            'avg_rating': round(topN[1].avg_rating,1),
            'movieid': topN[1].movieid
            },
            {
            'thumbnail': topN[2].thumbnail,
            'watchlink': topN[2].watchlink,
            'avg_rating': round(topN[2].avg_rating,1),
            'movieid': topN[2].movieid
            },
            {
            'thumbnail': topN[3].thumbnail,
            'watchlink': topN[3].watchlink,
            'avg_rating': round(topN[3].avg_rating,1),
            'movieid': topN[3].movieid
            },
            {
            'thumbnail': topN[4].thumbnail,
            'watchlink': topN[4].watchlink,
            'avg_rating': round(topN[4].avg_rating,1),
            'movieid': topN[4].movieid
            }
            ]
            session[data['user']] = {
                'which_movie':int(data['movie']),
                'recommendations':recommendations
            }
        Kb.clear_session()
    return redirect(url_for('refresh_recommendations'))

@app.route('/refresh_recommendations')
def refresh_recommendations():
    return render_template('refresh_recommendations.html')


#Data Migration Script=========================================
# @app.route('/Interactions_migrated', methods = ['GET','POST'])
# def migratedInteractions():
#     status = migrate_interactions()
#     return render_template('Interactions_migrated.html', status = status)

# @app.route('/Users_migrated', methods = ['GET','POST'])
# def migratedUsers():
#     status = migrate_customers()
#     return render_template('Users_migrated.html', status = status)

# @app.route('/Movies_migrated', methods = ['GET','POST'])
# def migratedMovies():
#     status = migrate_movies()
#     return render_template('Movies_migrated.html', status = status)

#================================================================


