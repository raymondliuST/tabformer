from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('abc').getOrCreate()


df = spark.read.parquet("data/people-model-20240124.parquet").toPandas()

col = ['estid', 'dayOfWeek', 'dayOfMonth', 'timeOfDay', 'url_category', 'domain_category', 'os', 'deviceType', 'li_age', 'li_gender', 'li_income', 'city_location', 'city_size', 'city_tech', 'city_urban', 'organization'] 

event_col = ['estid', 'dayOfWeek', 'dayOfMonth', 'timeOfDay', 'url_category', 'domain_category'] 
user_col = ['estid', 'os', 'deviceType', 'li_age', 'li_gender', 'li_income', 'city_location', 'city_size', 'city_tech', 'city_urban', 'organization'] 

event_data = df[event_col]
user_data = df[user_col]

event_data.to_csv("data/event/event_data.csv", index=False)
user_data.to_csv("data/user/user_data.csv", index=False)

df.to_csv("data/people-model-20240124.csv", index=False)
import pdb

pdb.set_trace()
