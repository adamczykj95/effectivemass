import boto3    
s3 = boto3.resource('s3')
bucket = s3.Bucket('bucketeer-88c06953-e032-4084-8845-f22694bbd8b4')
# suggested by Jordon Philips 
bucket.objects.all().delete()
