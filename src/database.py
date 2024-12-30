import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore 

cred = credentials.Certificate("../attentionindex-firebase-adminsdk-x5zeb-0f8634b6b9.json")
class Database:
  def __init__(self, meeting_name):
    self.firebase = firebase_admin.initialize_app(cred)
    self.db = firebase_admin.firestore.client(app = self.firebase)
    self.meeting_name = meeting_name

  def send(self,message : dict):
    self.db.collection(self.meeting_name).add(message)
    print('Pushed to database')

