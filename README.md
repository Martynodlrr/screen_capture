.env contains roboflow api key as API_KEY and debug as DEBUG

built into an exe from running
```cmd
pyinstaller main.spec
```
in cmd screen_captrue dir

To run: download as a zip, extract, delete everything except \build dir inside of \dist
double click on main.exe

To stop:
crtl+alt+f12

For debug file download from debug branch, same process as normal but a logging debug file will be created next to exe

this project uses the [valoAccuracy](https://universe.roboflow.com/martynodlrr/valoaccuracy) model from RoboFlow
