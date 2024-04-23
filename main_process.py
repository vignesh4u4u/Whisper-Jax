from flask import Flask,request
from gevent.pywsgi import WSGIServer
from main_code import openchat_code

app = Flask(__name__)
@app.route("/ml-ser" , methods =["POST"])
def text():
    try:
        if "query " not in request.form:
            return "please give the input query"
        else:
            query = request.form["query"]
            generate = openchat_code.text_generation(query)
            return generate
    except Exception as e:
        return ({"error":str(e)})

if __name__ == "__main__":
    print("Starting the server on port 8080")
    #flask_app.run(debug=False, host="0.0.0.0", port=8080)
    http_server = WSGIServer(('0.0.0.0', 8080),app)
    print('Server running on http://0.0.0.0:8080')
    http_server.serve_forever()