Notes:
 - run on windows 
 - install all requirements.txt
 - all py scripts require rplidar.py (needs to be in same folder)
 - documentation here: 

https://bucket-download.slamtec.com/3c1b2ba89f2a553f03893768648660b7528f779f/LM310_SLAMTEC_rplidarkit_usermanual_A2%20series_v1.0_en.pdf


eecs_env\Scripts\activate


ToDo:
- make sure all polar interpretations use the same coordinate system as the sensor (counterclockwise with North = 0 degrees)
- speed up accumulator implementation (use adjacency list instead of sparce array)

Done
- dealing with positive and negative valued windows for hough transform