from flask import Flask, render_template, request

app = Flask(__name__)
messages = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form.get('username')
        message = request.form.get('message')
        messages.append(f"{username}: {message}")
    return render_template('chat.html', messages=messages)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
