import pandas as pd
from predict_salary import predict_salary
def prepare_input(title, agency, location, experience, job_category='General'):
    return pd.DataFrame([{
        "Business Title": title,
        "Agency": agency,
        "Work Location": location,
        "Job Category": job_category,
        "Years of Experience": experience
    }])
