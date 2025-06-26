# AI-FraudCall-Detector
Visit the Live App at https://frauddetectionapp.publicvm.com/

Watch the Demo Video here https://drive.google.com/file/d/1IXspU9Jvv78TztyarFU5Apk1D78hK6dI/view?usp=share_link


## Steps for local deployment
Ensure you have docker installed

Run the qdrant DB 

`sudo docker run -d --name qdrant -d -p 6333:6333 qdrant/qdrant`

Create the DB collections

`python db_creation.py`

Run the Backend API

`python main.py`

Run the Streamlit UI

`streamlit run Home.py`
