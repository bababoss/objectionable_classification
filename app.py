from flask import Flask, request, jsonify, Response
import flask
import io
import json
import requests
import time

 
# initialize our Flask application and Redis server
app = flask.Flask(__name__)

 
# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
  return """
  Wrong URL!
  <pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
  return """
  An internal error occurred: <pre>{}</pre>
  See logs for full stacktrace.
  """.format(e), 500


@app.route("/api", methods=["POST"])
def predict():

    req_data=request.get_json()
    text_data  = req_data["text"].encode('utf-8')
    output_data={"text_data":text_data}
    response=jsonify(output_data)   
    response.status_code=200
    #return the data dictionary as a JSON response
    return response#flask.jsonify(output_data)



@app.route('/', methods=['POST'])
def api():
    if 'text' in request.json:
        print("[INFO] request recieved")

        # get the input text data
        # from the request header
        req_data=request.get_json()
        text_data  = req_data["text"].encode('utf-8')

        output_data={"text_data":text_data}

        # jsonify and send response
        # also set status code
        response=jsonify(output_data)   
        response.status_code=200


    else:
        response=Response(status=100)
    return response


if __name__ == '__main__':

      app.run(host='0.0.0.0', port = 5000, threaded=True)
      # setting debug = True is good for debugging, but it results in the model 
      # being loaded twice. There is no easy way to get rid of this without 
      # sacrificing auto reload which is useful in developement. 
      # Remove debug=True in  production.  