Corner Matching:

a SLAM system which utilizes the SLAMTEC RPLidar A2M8 to identify landmarks (90 degree corners) in an indoor environment and measure the change in position and bearing of the sensor over time.


To Run (windows): 
 - install pip
 - create a new virtual environment
    - pip install virtualenv
    - virtualenv venv 
    - venv\Scripts\activate
 - to exit virtual environment:
    - deactivate
 - run pip install -r requirements.txt
 - find the COM port connected to RPLidar in windows device manager


Notes:
 - all py scripts require rplidar.py (needs to be in same folder)
 - RPlidar documentation here: https://bucket-download.slamtec.com/3c1b2ba89f2a553f03893768648660b7528f779f/LM310_SLAMTEC_rplidarkit_usermanual_A2%20series_v1.0_en.pdf