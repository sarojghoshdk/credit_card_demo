from pymongo.mongo_client import MongoClient
import pandas as pd 
import json  # To store the data in mongodb in json form so we convert our data into json form

# uniform resource identifier
uri = "mongodb+srv://sarojghoshdk:sarojghoshdk@cluster0.wur1oso.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name & collection name
DATABASE_NAME = "credit_card_project"
COLLECTION_NAME = "credit_card_default"

# read the data as a dataframe
df = pd.read_csv(r"D:\Project\credit_card_default_prediction\notebook\creditCardFraud_28011964_120214.csv")
#df=df.drop("Unnamed: 0",axis=1)

# convert the data in json form
json_record = list(json.loads(df.T.to_json()).values())

# now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
