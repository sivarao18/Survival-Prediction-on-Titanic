from flask import Flask,render_template,request
import pickle

model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def homePage():
    return (render_template('index.html'))

@app.route('/predict',methods=['POST'])
def collectData():
    Pid=float(request.form['Pid'])
    pclass=float(request.form['pclass'])
    Sex=float(request.form['Sex'])
    Age=float(request.form['Age'])
    Sibsp=float(request.form['Sibsp'])
    Parch=float(request.form['Parch'])
    Fare=float(request.form['Fare'])
    Embarked=float(request.form['Embarked'])
    print(Pid,pclass,Sex,Age,Sibsp,Parch,Fare,Embarked)
    result=model.predict([[Pid,pclass,Sex,Age,Sibsp,Parch,Fare,Embarked]])
    return(str(result[0]))

if __name__=="__main__":
    app.run(debug=True)