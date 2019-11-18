from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,BooleanField,SubmitField, SelectField
from wtforms.validators import ValidationError, DataRequired,Email,EqualTo
from app import app
from app.models import Users
class LoginForm(FlaskForm):
    username = StringField('Username', validators = [DataRequired()])
    password = PasswordField('Password',validators = [DataRequired()])
    remember_me = BooleanField('Remember me?')
    submit = SubmitField('Sign In')

    # def search_username(self, username):
    #     user = Users.query.filter_by(username = username.data).first()
    #     if user is None:
    #         raise ValidationError('Enter valid username!')
    
    # def search_password(self, password):
    #     user = Users.query.filter_by(username = username.data).first()
    #     if user is None:
    #         raise ValidationError('Enter valid username!')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators = [DataRequired(message='Username cannot be left blank')])
    email = StringField('Email', validators = [DataRequired(message='Email cannot be left blank'), Email()])
    password = PasswordField('Password',validators = [DataRequired(message='Password cannot be left blank')])
    confirmPassword = PasswordField('Confirm Password',validators = [DataRequired(message='Confirm password cannot be left blank'),EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = Users.query.filter_by(username = username.data).first()
        if user is not None:
            raise ValidationError('Username already registered!')
            

    def validate_email(self, email):
        user = Users.query.filter_by(email = email.data).first()
        if user is not None:
            raise ValidationError('Email Id already registered!')


class SearchForm(FlaskForm):
    search = StringField('Search for movies...')
    submit = SubmitField('Search')
            



