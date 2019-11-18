from datetime import datetime
from flask_login import UserMixin
from app import db, login

class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, index = True, primary_key = True)
    userid = db.Column(db.Integer, index = True)
    username = db.Column(db.String(64), index = True)
    email = db.Column(db.String(120), index = True)
    password = db.Column(db.String(15))
    #user = db.relationship('Interactions', backref = 'user', lazy = 'dynamic')

class Movies(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    movieid = db.Column(db.Integer, index = True)
    moviename = db.Column(db.String(64))
    genre = db.Column(db.String(256))
    thumbnail = db.Column(db.String(64))
    watchlink = db.Column(db.String(256))
    avg_rating = db.Column(db.Float)
    N_ratings = db.Column(db.Integer)
    #movie = db.relationship('Interactions', backref = 'movie', lazy = 'dynamic')

class Interactions(db.Model):
    id = db.Column(db.Integer, index = True, primary_key = True)
    userid = db.Column(db.Integer, db.ForeignKey('users.userid'))
    movieid = db.Column(db.String(120), db.ForeignKey('movies.movieid'))
    rating = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, index = True, default = datetime.utcnow)

class Validations(db.Model):
    id = db.Column(db.Integer, index = True, primary_key = True)
    userid = db.Column(db.Integer, db.ForeignKey('users.userid'))
    movieid = db.Column(db.String(120), db.ForeignKey('movies.movieid'))
    rating = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, index = True, default = datetime.utcnow)

@login.user_loader
def load_user(id):
    return Users.query.get(int(id))
