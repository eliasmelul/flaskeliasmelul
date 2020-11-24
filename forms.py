from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectMultipleField, FieldList, SelectField, IntegerField
from projects.datasupply import get_cities, get_stock_tickers

#Forms
class RecommenderForm(FlaskForm):
    all_cities = get_cities()
    cities = SelectMultipleField("", choices=all_cities)
    submit = SubmitField('Submit')


class Finance_Basic(FlaskForm):
    stocks = get_stock_tickers()
    ticker = SelectMultipleField("", choices=stocks)
    days = IntegerField("Days to forecast")
    fsubmit = SubmitField('Submit')
    