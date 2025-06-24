# AI-FraudCall-Detector
Live App available at https://frauddetectionapp.publicvm.com/


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
