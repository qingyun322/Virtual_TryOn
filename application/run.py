#!/usr/bin/env python
import sys
sys.path.append('/home/ubuntu/Insight_Project/Virtural_Tryon')
from flaskexample import app
app.run(host = '0.0.0.0', debug = True)
