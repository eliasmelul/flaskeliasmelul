from flask import Flask, render_template, url_for, request
from projects.recommendersys import input_cities
from projects.finance import multiple_monte_carlo
from forms import RecommenderForm, Finance_Basic
import pandas as pd
from plotly.offline import plot


application = app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

# Routes
@app.route('/', methods=['GET','POST'])
def index():

    cities = False
    recommendations = False
    rec_form = RecommenderForm()
    fin_form = Finance_Basic()

    if rec_form.validate_on_submit() and rec_form.submit.data:
        cities = rec_form.cities.data 
        recommendations = input_cities(cities)
        rec_form.cities.data = []

    if fin_form.validate_on_submit() and fin_form.fsubmit.data:
        stock = fin_form.ticker.data 
        days = fin_form.days.data
        html, info = multiple_monte_carlo(stock, days)
    else:
        fin_form.ticker.data = ['BRK.B', 'NVDA','BNTX']
        fin_form.days.data = 252
        html, info = multiple_monte_carlo(['BRK.B', 'NVDA','BNTX'], 252)
    # if request.method == 'POST' and request.form.get('stockbutton') == 'Submit':
    #     print(request.form)
    #     u = request.form.get('states')
    #     print(u)
    

    return render_template('index.html', rec_form=rec_form, cities=cities, 
                            recommendations=recommendations, fin_form=fin_form, 
                            html=[html], info=info)

@app.route('/marketing', methods=['GET','POST'])
def marketing_crm():
    # Placeholder for CRM project site
    return render_template('random.html')

@app.route('/cityrecommender', methods=['GET', 'POST'])
def cityrecommender():
    # Placeholder for city recommender project site
    pass 

@app.route('/finance', methods=['GET', 'POST'])
def finance():
    # Placeholder for finance project site
    pass

if __name__ == "__main__":
    app.run(debug=True)
