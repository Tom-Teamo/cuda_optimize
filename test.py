import requests
import json
from bs4 import BeautifulSoup

session = requests.Session()

url = "https://connect.garmin.cn/modern/stress/2024-02-29"
headers = {"authorization": "bearer eyJyZWZyZXNoVG9rZW5WYWx1ZSI6ImEzMThlZDBkLWY3ZWMtNGRhMi05M2U0LTBkMTQxMjY5YTBjZiIsImdhcm1pbkd1aWQiOiJkYTA0Mzk4NS02NjczLTQwOWUtODMwNy1lZTZmNDUwMzdkNTIifQ=="}

r = session.get(url=url, headers=headers)

if r.status_code == 200:

    url_file = "https://connect.garmin.cn/wellness-service/wellness/dailyStress/2024-02-29"
    headers = {"authorization": session.auth}
    print(session.auth)
    visualization_response = session.get(url_file, headers=headers)
    
    if visualization_response.status_code == 200:
        print(visualization_response)
        
    else:
        print('Failed to retrieve visualization data:', visualization_response.status_code)
else:
    print("status code error:", r.status_code)
session.close()