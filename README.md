# Prediction of cardiac events from polarmaps
We train a CNN with three inputs that are later merged and produce a single prediction of a cardiac event. The three inputs are the cardiac polarmaps in rest, stress, and reserve states.
![polar_maps_demo](images/CNN_API4.gif?raw=true)   


### Old instructions to install virtualenv with tensorflow
virtualenv -p python3 tfvenv
source tfvenv/bin/activate
pip3 install --upgrade tensorflow
pip3 install keras
pip3 install flask
pip3 install Pillow

To run:
python keras_REST_API.py
To test on image:
<on another terminal> 
curl -X POST -F image=@rest0118.jpg 'http://127.0.0.1:5000/predict'

