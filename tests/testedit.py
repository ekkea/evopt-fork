import argparse
import json
from pathlib import Path

#import numpy

from evopt.app import app

parser = argparse.ArgumentParser(prog="testedit")
parser.add_argument("action", choices=["create", "update", "run"], help="")
parser.add_argument("file", type=str, default="", help="")
args = parser.parse_args()
print(args)
action = args.action
file = args.file

client = app.test_client()

if action=="create":
    file_path=Path(file)
    if file_path.exists():
        request = json.loads(file_path.read_text())
        response = client.post("/optimize/charge-schedule", json=request)
        test_case = {}
        test_case['request']=request
        test_case['response']=response.get_json()
        print(test_case)
        json.dump(test_case, fp=open('tmp/test_case.json',"w"))



#request = test_data["request"]
# response = test_data.get("expected_response")
