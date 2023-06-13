# deployment-recommendation-system

gcloud builds submit --tag gcr.io/rentaku-capstone/recomsys

gcloud run deploy --image gcr.io/rentaku-capstone/recomsys --platform managed
