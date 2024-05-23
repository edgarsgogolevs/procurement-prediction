from os import environ
import logging

from flask import Flask, render_template, request
from flask_caching import Cache

from modules.classifier import Classifier

classifier = Classifier()

root_lg = logging.getLogger()
logging.basicConfig(level=environ.get("LOGLEVEL", "INFO"))

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})

app = Flask(__name__)
cache.init_app(app)

@app.route("/")
def home():
    if "regNo" in request.args:
        regNo = request.args["regNo"]
        if regNo in classifier.company_profiles:
            recommendations = classifier.get_recommendations(regNo, n_recommendations=10)
            return render_template("index.html",
                                   recommendations=recommendations, regNo=regNo)
        return render_template("index.html", showError=True)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=int(environ.get("PORT", 8069)))

