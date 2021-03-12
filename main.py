from flask import Flask, send_file
from flask_restful import Resource, Api
from ml import do_ml

app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def get(self):
        return {

        }


class Dashboard(Resource):
    def get(self):
        bytes_obj = do_ml()

        return send_file(bytes_obj,
                         attachment_filename='plot.png',
                         mimetype='image/png')



api.add_resource(Prediction, '/prediction')
api.add_resource(Dashboard,
                 '/dashboard',
                 '/dashboard/feature-importance',
                 '/dashboard/confusion-matrix')

if __name__ == '__main__':
    app.run(debug=True)

# TODO:
#   [x]Make a correlation graph between selected features
#   [x]Explore RFE
#   [x]Review & implement cross validation
#   [x]Add column names in importance graph
#   [x]Learn the basics of plt
#   [x]Review and implement confusion matrix (evaluation metrics)
#   [x]Create summary statistics of accuracy, precision, recall, f-score
#   [x]Decide whether to use the dataset
#   [x]Calculate price var based on a fixed amount investment divided equally into winning picks
#   [x]Investigate class, price var % mismatch in pl_df (index f-d up during transformation, dataframing?)
#   [x]Fix buy/skip column of pl_df
#   [x]Try using aggregated dataset from year 2014-2018 and see if model improves
#   [x]Fine tune algorithm(tune parameters, missing data threshold) to boost ROI and beat S&P500(y2019 ~28%)
#   [x]Get Names of the true positive stocks
#   [x]See if k-fold method increases performance
#   [x]See if feature correlation heatmap works
#   [x]Decide whether to use fewer features for a better correlation heatmap
#   []Put feature names on corr heatmap axis, align axis
#   [x]Draft the capstone topic approval form
#   []Create stock-ml api with flask
