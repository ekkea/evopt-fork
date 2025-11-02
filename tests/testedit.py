import argparse

import json
from pathlib import Path

import numpy
#import pytest

from evopt.app import app

parser = argparse.ArgumentParser(prog="testedit")
parser.add_argument("action", choices=["create", "update", "run"], help="")
parser.add_argument("file", type=str, default="", required=False, help="")
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
        


#request = test_data["request"]
# response = test_data.get("expected_response")
