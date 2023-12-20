import pickle
from flask import Flask, render_template, request

app = Flask("predicting lumpy skin disease")

# Load the trained random forest classifier model outside of the perform_eda function
rf_classifier_model = pickle.load(open('rf_classifier_model.pkl', 'rb'))


def make_prediction(model, features):
    # Assuming features is a list of input values in the order expected by your model
    prediction = model.predict([features])
    return prediction[0]


@app.route("/")
def index():
    return render_template("first_page.html")


@app.route("/form")
def form_page():
    return render_template("form_page.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "POST":
        # Get the geographical details from the form
        monthly_cloud_cover = request.form["monthly_cloud_cover"]
        diurnal_temperature_range = request.form["diurnal_temperature_range"]
        frost_day_frequency = request.form["frost_day_frequency"]
        potential_evapotranspiration = request.form["potential_evapotranspiration"]
        precipitation = request.form["precipitation"]

        # Make a prediction using the machine learning model
        prediction = make_prediction(rf_classifier_model, [monthly_cloud_cover, diurnal_temperature_range, frost_day_frequency, potential_evapotranspiration, precipitation])

        # Redirect the user to the prediction result page
        return render_template("predict_page.html", prediction=prediction)
    else:
        # Handle the GET request
        monthly_cloud_cover_param = request.args.get("monthly_cloud_cover")

        if monthly_cloud_cover_param is not None:
            try:
                monthly_cloud_cover = float(monthly_cloud_cover_param)
            except ValueError:
                # Handle the case where the conversion to float fails
                return render_template("error_page.html", error_message="Invalid value for monthly cloud cover.")
        else:
            # Handle the case where the parameter is not present
            return render_template("error_page.html", error_message="Monthly cloud cover parameter is missing.")

        # Continue with the rest of your GET request handling
        return render_template("predict_page.html", prediction=None)



def perform_eda():
    # Import necessary libraries
    import pandas as pd
    import numpy as np

    # Load the dataset
    data = pd.read_csv('Lumpy skin disease data.csv')

    # Descriptive statistics for each factor
    monthly_cloud_cover_stats = data['monthly_cloud_cover'].describe()
    diurnal_temperature_range_stats = data['diurnal_temperature_range'].describe()
    frost_day_frequency_stats = data['frost_day_frequency'].describe()
    potential_evapotranspiration_stats = data['potential_evapotranspiration'].describe()
    precipitation_stats = data['precipitation'].describe()

    # Correlation matrix between factors
    correlation_matrix = data.corr()

    # Prepare data for the template
    monthly_cloud_cover_distribution = monthly_cloud_cover_stats.to_dict()
    # ...

    return {
        'monthly_cloud_cover_distribution': monthly_cloud_cover_distribution,
        # ...
    }


@app.route("/eda")
def eda_page():
    # Perform EDA analysis and prepare data for the template
    eda_results = perform_eda()

    return render_template("eda_page.html", eda_results=eda_results)


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":

    app.run(debug=True)