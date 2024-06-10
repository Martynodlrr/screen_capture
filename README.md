To run: download as a zip, extract, delete everything except \build dir inside of \dist
double click on main.exe

To stop:
crtl+alt+f12

To run with your own api key (or if mine is maxed out lol)
create a .env containing roboflow api key as API_KEY and debug as DEBUG (True or False) in the dir next to main.py

built into an exe from running
```cmd
pyinstaller main.spec
```
in cmd screen_captrue dir

For debug file download from debug branch, same process as normal but a logging debug file will be created next to exe

this project uses the [valoAccuracy](https://universe.roboflow.com/martynodlrr/valoaccuracy) model from RoboFlow
